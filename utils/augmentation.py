"""缺失模拟：整段丢弃、与附件3同分布的零散短游程、以及多尺度连续缺失区间。

三种形态各有用途：

1. ``whole``：整段丢弃模态，覆盖“模态完全不可用”的极端压力测试。
2. ``local``：语音与视觉在**相同词位**上出现长度 1-4 位的多段零散短游程。
   附件3（对齐版 30 条）实测即为此形态：文本从不缺失、两模态缺失位置
   在 29/30 条一致、额外缺失率 0%-46%（合计约 22%）。
3. ``interval``：在长度为 L 的有效序列上生成连续缺失区间 [s, s+ℓ)，其中
   ℓ/L 从候选比例 {0.1, 0.2, 0.4, 0.6} 中抽取。模态组合覆盖单模态、双模态
   与三模态，且多模态缺失区间可以部分重叠。这些比例是**建议的实验设置**，
   不是附件3 的已知缺失比例，目的是让模型同时见到轻度与严重缺失。

默认 ``pattern="auto"`` 按 ``whole_probability`` 混合三者。所有操作只改动
训练输入副本，不修改原始样本与真实标签。
"""

from __future__ import annotations

import random

import torch

MODALITIES = ("text", "audio", "vision")
PATTERNS = ("auto", "whole", "local", "interval", "mixed")
_LOCAL_RUN_MAX = 4  # 附件3实测单段游程长度不超过4位。
_LOCAL_ATTEMPTS = 200
INTERVAL_RATIOS = (0.1, 0.2, 0.4, 0.6)
# 模态组合及其抽样权重：语音+视觉最贴近专项数据，同时保留单模态与含文本的组合。
INTERVAL_PLANS = (
    (("audio",), 1.0),
    (("vision",), 1.0),
    (("text",), 0.5),
    (("audio", "vision"), 2.0),
    (("text", "audio"), 0.5),
    (("text", "vision"), 0.5),
    (("text", "audio", "vision"), 0.5),
)
_CLONE_KEYS = ("tokens", "audio", "vision", "text_mask", "audio_mask", "vision_mask")


def _positions(valid: torch.Tensor) -> torch.Tensor:
    return torch.where(valid)[0]


def local_drops(valid: torch.Tensor, rate: float, location: str | None,
                rng: random.Random) -> torch.Tensor:
    """在有效位上取多段连续短游程，返回被遮蔽的位置索引。"""
    index = _positions(valid)
    total = int(index.numel())
    if total == 0 or rate <= 0:
        return index[:0]
    target = max(1, min(total, round(rate * total)))
    if location == "start":
        low, high = 0, max(1, total // 3)
    elif location == "middle":
        low, high = total // 3, max(total // 3 + 1, (2 * total) // 3)
    elif location == "end":
        low, high = max(0, (2 * total) // 3), total
    elif location in (None, "random"):
        low, high = 0, total
    else:
        raise ValueError(f"Unknown location: {location}")
    chosen: set[int] = set()
    for _ in range(_LOCAL_ATTEMPTS):
        if len(chosen) >= target:
            break
        length = rng.randint(1, _LOCAL_RUN_MAX)
        start = rng.randint(low, max(low, high - 1))
        chosen.update(range(start, min(start + length, high)))
    return index[torch.tensor(sorted(chosen), dtype=torch.long)]


def interval_drops(valid: torch.Tensor, ratio: float, location: str | None,
                   rng: random.Random, anchor_fraction: float | None = None) -> torch.Tensor:
    """在有效位上取一段连续缺失区间，长度约为 ``ratio`` × 有效长度。

    ``anchor_fraction`` 给出区间中心在有效区间内的相对位置（0-1），用于让
    多个模态的缺失区间**部分重叠**；为 None 时按 ``location`` 放置。
    """
    index = _positions(valid)
    total = int(index.numel())
    if total == 0 or ratio <= 0:
        return index[:0]
    length = max(1, min(total, round(ratio * total)))
    span = total - length
    if anchor_fraction is not None:
        start = int(round(anchor_fraction * span))
    elif location == "start":
        start = 0
    elif location == "middle":
        start = span // 2
    elif location == "end":
        start = span
    elif location in (None, "random"):
        start = rng.randint(0, span) if span > 0 else 0
    else:
        raise ValueError(f"Unknown location: {location}")
    start = max(0, min(start, span))
    return index[start:start + length]


def _sample_plan(rng: random.Random) -> tuple[str, ...]:
    total = sum(weight for _, weight in INTERVAL_PLANS)
    target = rng.random() * total
    for names, weight in INTERVAL_PLANS:
        target -= weight
        if target <= 0:
            return names
    return INTERVAL_PLANS[-1][0]


def _apply_whole(out: dict, masks: dict, row: int, names: tuple[str, ...]) -> None:
    for name in names:
        masks[name][row].zero_()
        if name == "text":
            out["tokens"][row, 0].zero_()
        else:
            out[name][row].zero_()


def _apply_drops(out: dict, masks: dict, row: int, names: tuple[str, ...],
                 drops: torch.Tensor) -> None:
    if drops.numel() == 0:
        return
    for name in names:
        masks[name][row, drops] = False
        if name == "text":
            out["tokens"][row, 0, drops] = 0
        else:
            out[name][row, drops] = 0.0


def mask_batch(
    batch: dict[str, torch.Tensor],
    *,
    seed: int,
    probability: float = 0.75,
    missing_modalities: tuple[str, ...] | None = None,
    local_rate: float | None = None,
    location: str | None = None,
    pattern: str = "auto",
    ratios: tuple[float, ...] = INTERVAL_RATIOS,
    interval_modalities: tuple[str, ...] | None = None,
    overlap_probability: float = 0.5,
    whole_probability: float = 0.3,
    local_rate_range: tuple[float, float] = (0.05, 0.45),
) -> dict[str, torch.Tensor]:
    """按指定形态或 mixed/auto 形态遮蔽模态。

    - ``missing_modalities``：整段丢弃这些模态（优先级最高）。
    - ``local_rate``：在语音与视觉的相同词位上制造多段短游程。
    - ``location``：``start`` / ``middle`` / ``end`` / ``random``，控制区间或游程位置。
    - ``pattern``：``whole`` / ``local`` / ``interval`` / ``mixed``；``auto`` 时先按
      ``probability`` 判定是否遮蔽，再以 ``whole_probability`` 抽取整段形态，
      其余概率在局部短游程与连续区间之间均分。
    """
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if not 0 <= whole_probability <= 1:
        raise ValueError("whole_probability must be in [0, 1]")
    if not 0 <= overlap_probability <= 1:
        raise ValueError("overlap_probability must be in [0, 1]")
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}")
    if local_rate is not None and not 0 <= local_rate <= 1:
        raise ValueError("local_rate must be in [0, 1]")
    if any(not 0 < ratio <= 1 for ratio in ratios):
        raise ValueError("ratios must be in (0, 1]")
    if local_rate is not None and missing_modalities is not None:
        raise ValueError("pass either missing_modalities or local_rate, not both")
    for names in (missing_modalities, interval_modalities):
        if names is not None and (not set(names) <= set(MODALITIES)
                                  or len(set(names)) != len(names)):
            raise ValueError("modality names must be unique and within text/audio/vision")

    rng = random.Random(seed)
    out = batch.copy()
    for key in _CLONE_KEYS:
        out[key] = batch[key].clone()
    masks = {name: out[f"{name}_mask"] for name in MODALITIES}
    remainder = (1.0 - whole_probability) / 2.0

    for row in range(len(out["tokens"])):
        if missing_modalities is not None:
            _apply_whole(out, masks, row, missing_modalities)
            continue
        if local_rate is not None:
            drops = local_drops(masks["audio"][row] | masks["vision"][row], local_rate, location, rng)
            _apply_drops(out, masks, row, ("audio", "vision"), drops)
            continue
        if rng.random() >= probability:
            continue

        family = pattern
        if family == "auto":
            draw = rng.random()
            if draw < whole_probability:
                family = "whole"
            elif draw < whole_probability + remainder:
                family = "local"
            else:
                family = "interval"
        elif family == "mixed":
            family = "whole" if rng.random() < whole_probability else "local"

        if family == "whole":
            _apply_whole(out, masks, row, ("audio", "vision"))
        elif family == "local":
            rate = rng.uniform(*local_rate_range)
            drops = local_drops(masks["audio"][row] | masks["vision"][row], rate, location, rng)
            _apply_drops(out, masks, row, ("audio", "vision"), drops)
        else:
            names = tuple(interval_modalities) if interval_modalities else _sample_plan(rng)
            anchor = rng.random() if (len(names) > 1 and rng.random() < overlap_probability) else None
            for name in names:
                drops = interval_drops(masks[name][row], rng.choice(ratios), location, rng, anchor)
                _apply_drops(out, masks, row, (name,), drops)
    return out


def drop_modalities(batch: dict[str, torch.Tensor],
                    names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """永久遮蔽指定模态，用于训练单模态基线模型。"""
    if not set(names) <= set(MODALITIES):
        raise ValueError(f"unknown modality in {names}")
    out = batch.copy()
    for key in _CLONE_KEYS:
        out[key] = batch[key].clone()
    for name in names:
        out[f"{name}_mask"].zero_()
        if name == "text":
            out["tokens"][:, 0].zero_()
        else:
            out[name].zero_()
    return out


def _demo_batch(rows: int = 2) -> dict[str, torch.Tensor]:
    valid = torch.zeros(rows, 50, dtype=torch.bool)
    valid[:, 2:42] = True  # 模拟附件2的有效区间：首末两位为固有零位。
    return {
        "tokens": torch.ones(rows, 3, 50, dtype=torch.long),
        "audio": torch.ones(rows, 50, 74),
        "vision": torch.ones(rows, 50, 35),
        "text_mask": torch.ones(rows, 50, dtype=torch.bool),
        "audio_mask": valid.clone(),
        "vision_mask": valid.clone(),
    }


def _run_lengths(flags) -> list[int]:
    lengths, run = [], 0
    for flag in flags:
        if flag:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def demo() -> None:
    batch = _demo_batch()
    valid_total = int(batch["audio_mask"].sum())
    original = {key: batch[key].clone() for key in _CLONE_KEYS}

    whole = mask_batch(batch, seed=7, probability=1.0, missing_modalities=("audio", "vision"))
    assert not whole["audio_mask"].any() and not whole["vision_mask"].any()
    assert whole["text_mask"].all() and whole["tokens"].equal(batch["tokens"])
    assert mask_batch(batch, seed=7, missing_modalities=("text",))["text_mask"].sum() == 0

    local = mask_batch(batch, seed=11, probability=1.0, local_rate=0.2)
    dropped_a = batch["audio_mask"] & ~local["audio_mask"]
    dropped_v = batch["vision_mask"] & ~local["vision_mask"]
    assert torch.equal(dropped_a, dropped_v), "音频与视觉必须遮蔽相同词位"
    assert local["text_mask"].all() and local["tokens"].equal(batch["tokens"])
    realized = int(dropped_a.sum()) / valid_total
    assert 0.10 < realized < 0.32, f"realized local rate out of range: {realized}"
    lengths = _run_lengths(dropped_a[0].tolist())
    assert lengths and max(lengths) <= _LOCAL_RUN_MAX, f"runs too long: {lengths}"
    assert len(lengths) >= 2, f"expected several short runs, got {lengths}"
    assert mask_batch(batch, seed=11, probability=1.0, local_rate=0.2)["audio_mask"].equal(
        local["audio_mask"]), "same seed must reproduce"

    ends = mask_batch(batch, seed=5, probability=1.0, local_rate=0.3, location="end")
    assert int((batch["audio_mask"] & ~ends["audio_mask"])[:, :20].sum()) == 0

    # 连续缺失区间：长度必须等于 round(ratio × 有效长度)，且落点在指定位置。
    for ratio in INTERVAL_RATIOS:
        interval = mask_batch(batch, seed=3, probability=1.0, pattern="interval",
                              ratios=(ratio,), interval_modalities=("audio",), location="start")
        dropped = batch["audio_mask"] & ~interval["audio_mask"]
        expected = round(ratio * valid_total / batch["audio_mask"].shape[0])
        assert int(dropped.sum()) == expected * dropped.shape[0], (ratio, int(dropped.sum()), expected)
        for row in range(dropped.shape[0]):
            runs = _run_lengths(dropped[row].tolist())
            assert len(runs) == 1, f"interval must be one contiguous run: {runs}"
            assert int(dropped[row].nonzero()[0]) == 2, "location=start 的区间应从首个有效位开始"

    # 单模态 / 双模态 / 三模态都要出现，且重叠模式的区间必须有交集。
    plans, overlaps = set(), 0
    for seed in range(120):
        masked = mask_batch(batch, seed=seed, probability=1.0, pattern="interval", ratios=(0.4,))
        hit = tuple(name for name in MODALITIES
                    if int((batch[f"{name}_mask"] & ~masked[f"{name}_mask"]).sum()) > 0)
        plans.add(hit)
        if len(hit) > 1:
            first = batch[f"{hit[0]}_mask"] & ~masked[f"{hit[0]}_mask"]
            second = batch[f"{hit[1]}_mask"] & ~masked[f"{hit[1]}_mask"]
            if int((first & second).sum()) > 0:
                overlaps += 1
    assert any(len(hit) == 1 for hit in plans), f"缺少单模态缺失: {plans}"
    assert any(len(hit) == 2 for hit in plans), f"缺少双模态缺失: {plans}"
    assert any(len(hit) == 3 for hit in plans), f"缺少三模态缺失: {plans}"
    assert overlaps > 0, "重叠模式未产生交集"

    # 混合模式覆盖整段与部分缺失。
    mixed = [mask_batch(batch, seed=200 + i, probability=1.0) for i in range(24)]
    assert any(not m["audio_mask"].any() for m in mixed), "auto 模式必须产生整段缺失"
    partial = [int((batch["audio_mask"] & ~m["audio_mask"]).sum()) for m in mixed]
    assert any(0 < value < valid_total for value in partial), "auto 模式必须产生部分缺失"

    for key in _CLONE_KEYS:
        assert batch[key].equal(original[key]), f"mask_batch 修改了输入 {key}"
    assert int(batch["tokens"].min()) == 1, "真实标签与原始 token 不得被修改"


if __name__ == "__main__":
    demo()
    print("augmentation.py checks passed")
