"""问题三的语义闭合与时间映射规则。"""

from __future__ import annotations


ANSWER_OPENERS = {"absolutely", "certainly", "definitely", "yes", "no", "not", "never",
                  "however", "but", "actually", "of", "sure"}


def context_positions(mapping: dict, start: int, end: int) -> list[int]:
    """取核心词所在句的前文；简短回答还保留上一问句。"""
    positions = mapping.get("positions", [])
    words = mapping.get("words", [])
    core = [positions[i] for i in range(start, min(end, len(positions)))
            if positions[i] is not None]
    if not core:
        return []
    first = min(entry["word_index"] for entry in core)
    boundaries = [i for i in range(first)
                  if words[i]["word"].rstrip(' \"\'])}”').endswith((".", "?", "!"))]
    sentence_start = boundaries[-1] + 1 if boundaries else 0
    opener = words[first]["word"].strip('"“”.,!?').lower()
    if opener in ANSWER_OPENERS and boundaries:
        sentence_start = boundaries[-2] + 1 if len(boundaries) > 1 else 0
    core_words = {entry["word_index"] for entry in core}
    return [i for i, entry in enumerate(positions)
            if entry is not None and sentence_start <= entry["word_index"] < first
            and entry["word_index"] not in core_words]


def mapped_event(mapping: dict, positions_used: list[int]) -> dict:
    """只用实际匹配词位给出秒级区间，保留部分映射状态。"""
    positions = mapping.get("positions", [])
    matched = [positions[i] for i in positions_used
               if i < len(positions) and positions[i] is not None]
    if not matched:
        return {"start_seconds": "", "end_seconds": "", "mapping_status": "unmapped"}
    return {"start_seconds": min(word["start"] for word in matched),
            "end_seconds": max(word["end"] for word in matched),
            "mapping_status": "full" if len(matched) == len(positions_used) else "partial"}
