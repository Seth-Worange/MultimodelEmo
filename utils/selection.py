"""问题二验证集选模分数与缺失退化量。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


MAIN_VIEWS = ("clean", "local", "interval")
DEGRADATION_VIEWS = ("local", "interval")


def view_score(values: Mapping[str, float]) -> float:
    """单视图分数：Macro-F1 与按标签跨度 3 归一化的 MAE 等权。"""
    return 0.5 * (1.0 - float(values["macro_f1"])) + float(values["mae"]) / 6.0


def selection_score(view_metrics: Mapping[str, Mapping[str, float]],
                    views: Sequence[str] = MAIN_VIEWS) -> float:
    """仅对指定验证视图取均值；默认不纳入整模态缺失视图。"""
    if not views:
        raise ValueError("selection views must not be empty")
    return sum(view_score(view_metrics[view]) for view in views) / len(views)


def relative_degradation(view_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """报告局部缺失相对完整输入的 F1 与 MAE 变化。"""
    clean = view_metrics["clean"]
    return {
        view: {
            "delta_macro_f1": float(view_metrics[view]["macro_f1"] - clean["macro_f1"]),
            "delta_mae": float(view_metrics[view]["mae"] - clean["mae"]),
        }
        for view in DEGRADATION_VIEWS
    }
