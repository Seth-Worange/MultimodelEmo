"""语境否定族切片：短语族识别、语境对照组挖掘与切片掩码。

针对"Absolutely not."类断章取义错误：同一局部否定短语的极性由语境决定
（附件2实测同一短语族金标签正负几乎对半），模型易学成词汇捷径。本模块
只用附件2自身的 raw_text 与标签构造切片和对照组，不引入任何外部数据。

短语族划分（大小写不敏感，词边界匹配）：

- ``answer_neg``：应答式否定（absolutely/definitely/certainly/really not），
  单独看强负、在问答语境里常为正向安慰。
- ``strong_neg``：强否定（no way / not at all / by no means）。
- ``neg_idiom``：否定形式的正/中性习语（not bad / can't wait / nothing wrong）。
- ``not_only``：not only 递进修正，局部词面偏正、金标签两极都有。
"""

from __future__ import annotations

import re

import numpy as np

NEGATION_PATTERNS: dict[str, re.Pattern] = {
    "answer_neg": re.compile(
        r"\b(?:absolutely|definitely|certainly|really|totally|simply|utterly)\s+not\b", re.I),
    "strong_neg": re.compile(r"\b(?:no way|not at all|by no means|in no way)\b", re.I),
    "neg_idiom": re.compile(
        r"\b(?:not bad|can'?t wait|no problem|nothing wrong|nothing to complain|"
        r"doesn'?t (?:get|seem) any (?:better|worse)|no complaints)\b", re.I),
    "not_only": re.compile(r"\bnot only\b", re.I),
}

BOUNDARY_BAND = 0.5  # |y| <= 0.5 视为中性/弱情感边界带
STRONG_THRESHOLD = 2.0  # |y| >= 2 视为强情感


def negation_tags(raw_texts: list[str]) -> list[str | None]:
    """每条文本命中的短语族名；未命中为 None，多族命中取第一个命中的族。"""
    tags: list[str | None] = []
    for text in raw_texts:
        hit = None
        for name, pattern in NEGATION_PATTERNS.items():
            if pattern.search(text or ""):
                hit = name
                break
        tags.append(hit)
    return tags


def negation_family_mask(raw_texts: list[str]) -> np.ndarray:
    """布尔掩码：是否命中任一语境否定短语族。"""
    return np.array([tag is not None for tag in negation_tags(raw_texts)], dtype=bool)


def mine_context_pairs(raw_texts: list[str], classes: np.ndarray, sentiment: np.ndarray,
                       *, strength_gap: float = 1.5, max_pairs: int = 512) -> list[tuple[int, int]]:
    """挖掘"同短语族、语境相反"的样本对，作为语境对照组。

    收录规则：同一短语族内，两类极性相反（0 vs 2），或强度差 >= strength_gap。
    返回下标对 (i, j)，保证 sentiment[i] >= sentiment[j]，供排序损失使用。
    只使用传入切分自身的标签（训练时只传训练集），不涉及外部数据或伪标签。
    """
    classes = np.asarray(classes)
    sentiment = np.asarray(sentiment, dtype=np.float32)
    tags = negation_tags(raw_texts)
    by_family: dict[str, list[int]] = {}
    for index, tag in enumerate(tags):
        if tag is not None:
            by_family.setdefault(tag, []).append(index)
    pairs: list[tuple[int, int, float]] = []
    for members in by_family.values():
        for offset, i in enumerate(members):
            for j in members[offset + 1:]:
                opposite = {int(classes[i]), int(classes[j])} == {0, 2}
                gap = abs(float(sentiment[i] - sentiment[j]))
                if opposite or gap >= strength_gap:
                    a, b = (i, j) if sentiment[i] >= sentiment[j] else (j, i)
                    # 排序按强度差从大到小，截断时保留信息量最大的对。
                    pairs.append((a, b, gap))
    pairs.sort(key=lambda item: -item[2])
    return [(a, b) for a, b, _ in pairs[:max_pairs]]


def slice_masks(raw_texts: list[str], classes: np.ndarray, sentiment: np.ndarray) -> dict[str, np.ndarray]:
    """评估与加权用的标准切片掩码。"""
    classes = np.asarray(classes)
    sentiment = np.asarray(sentiment, dtype=np.float32)
    return {
        "negation": negation_family_mask(raw_texts),
        "boundary": np.abs(sentiment) <= BOUNDARY_BAND,
        "strong": np.abs(sentiment) >= STRONG_THRESHOLD,
        "neutral": classes == 1,
    }


def sample_weights(raw_texts: list[str], sentiment: np.ndarray, *,
                   negation_boost: float = 0.0, boundary_boost: float = 0.0) -> np.ndarray:
    """训练加权：在 1.0 的基础上对否定族与边界带样本加成。"""
    sentiment = np.asarray(sentiment, dtype=np.float32)
    weights = np.ones(len(sentiment), dtype=np.float32)
    if negation_boost:
        weights += np.float32(negation_boost) * negation_family_mask(raw_texts)
    if boundary_boost:
        weights += np.float32(boundary_boost) * (np.abs(sentiment) <= BOUNDARY_BAND)
    return weights


def report(raw_texts: list[str], classes: np.ndarray, sentiment: np.ndarray) -> str:
    """打印短语族 × 标签分布，供论文数据表与自检使用。"""
    tags = negation_tags(raw_texts)
    classes = np.asarray(classes)
    sentiment = np.asarray(sentiment, dtype=np.float32)
    lines = ["短语族,条数,负,中,正,y均值,y范围"]
    for name in list(NEGATION_PATTERNS) + ["(无命中)"]:
        hit = np.array([tag == name for tag in tags]) if name != "(无命中)" \
            else np.array([tag is None for tag in tags])
        if not hit.any():
            continue
        y = sentiment[hit]
        lines.append(",".join([
            name, str(int(hit.sum())),
            str(int((classes[hit] == 0).sum())), str(int((classes[hit] == 1).sum())),
            str(int((classes[hit] == 2).sum())), f"{y.mean():.2f}", f"{y.min():.2f}~{y.max():.2f}"]))
    pairs = mine_context_pairs(raw_texts, classes, sentiment)
    lines.append(f"语境对照组,{len(pairs)}对")
    return "\n".join(lines)


def demo() -> None:
    texts = [
        "If I blow it at the team exercise, should I kiss my chances of cheering \"GO BLUE\" goodbye?] Absolutely not.",
        "It's definitely not worth the money.",
        "There is nothing wrong with that.",
        "The movie was great.",
    ]
    classes = np.array([2, 0, 1, 2])
    sentiment = np.array([1.0, -2.0, 0.0, 2.0], dtype=np.float32)
    mask = negation_family_mask(texts)
    assert mask.tolist() == [True, True, True, False]
    tags = negation_tags(texts)
    assert tags[0] == "answer_neg" and tags[2] == "neg_idiom"
    pairs = mine_context_pairs(texts, classes, sentiment)
    assert (0, 1) in pairs, f"answer_neg 正负对照对必须被挖出: {pairs}"
    weights = sample_weights(texts, sentiment, negation_boost=1.0, boundary_boost=0.5)
    assert weights[0] == 2.0 and weights[1] == 2.0 and weights[2] == 2.5 and weights[3] == 1.0
    assert slice_masks(texts, classes, sentiment)["boundary"].tolist() == [False, False, True, False]
    print(report(texts, classes, sentiment))


if __name__ == "__main__":
    demo()
