"""描述词元时间映射的覆盖范围，不把词元匹配当作声学对齐验证。"""


def alignment_coverage(mapping: dict) -> dict:
    words = mapping.get("words", [])
    positions = [entry for entry in mapping.get("positions", []) if entry is not None]
    mapped_words = {entry["word_index"] for entry in positions}
    count = len(words)
    scope = "unavailable" if not count else (
        "aligned_transcript" if len(mapped_words) == count else "partial_aligned_transcript")
    return {"evidence_scope": scope,
            "mapped_word_count": len(mapped_words),
            "aligned_word_count": count,
            "aligned_word_coverage": len(mapped_words) / count if count else 0.0,
            "mapped_time_start": min((p["start"] for p in positions), default=None),
            "mapped_time_end": max((p["end"] for p in positions), default=None),
            "aligned_last_word_end": max((w["end"] for w in words), default=None)}


def mapped_word_text(positions: list[dict]) -> str:
    """合并同一词的子词，保留不同词位上重复出现的词。"""
    words = {}
    for entry in positions:
        words.setdefault(entry["word_index"], entry["word"])
    return " ".join(words.values())
