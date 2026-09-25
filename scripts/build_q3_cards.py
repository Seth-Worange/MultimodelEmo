"""把附件4现有预测与时间映射打包为离线解释卡页面。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "predictions_q3_mixed_neutral"
ALIGNMENT = ROOT / "outputs" / "runs" / "main" / "q3_alignment.json"
VIDEO_DIR = (ROOT / "data" / "附件4-可解释专项视频样本与特征文件"
             / "附件4-可解释专项视频样本与特征文件" / "未对齐版本" / "videos")
TEMPLATE = ROOT / "analysis" / "q3_cards_template.html"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def make_records() -> list[dict]:
    predictions = read_rows(OUTPUT / "q3_predictions.csv")
    evidence = read_rows(OUTPUT / "q3_selected_evidence.csv")
    alignment = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
    if len(predictions) != 20 or len(evidence) != 20:
        raise ValueError("附件4预测与证据必须各有20条")
    by_evidence = {row["id"]: row for row in evidence}
    records = []
    for row in predictions:
        sample_id = row["id"]
        if sample_id not in by_evidence or sample_id not in alignment:
            raise ValueError(f"缺少样本{sample_id}的证据或时间映射")
        path = VIDEO_DIR / f"{sample_id}.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(ROOT)
        video_url = "../../" + "/".join(quote(part) for part in relative.parts)
        records.append({
            "id": sample_id,
            "prediction": row,
            "evidence": by_evidence[sample_id],
            "alignment": {"words": alignment[sample_id]["words"],
                          "token_match_fraction": alignment[sample_id]["token_match_fraction"],
                          "status": alignment[sample_id]["status"]},
            "video_url": video_url,
        })
    return records


def main() -> None:
    records = make_records()
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace("&", "\\u0026")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__CARD_DATA__", payload)
    target = OUTPUT / "explanation_cards.html"
    target.write_text(html, encoding="utf-8")
    print(f"wrote {target} ({len(records)} samples, videos linked in place)")


if __name__ == "__main__":
    main()
