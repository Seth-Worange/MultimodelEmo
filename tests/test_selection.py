"""验证问题二选模规则与缺失退化量。"""

import unittest

from utils.selection import relative_degradation, selection_score, view_score


class SelectionTest(unittest.TestCase):
    def test_whole_does_not_change_main_score(self):
        values = {
            "clean": {"macro_f1": 0.6, "mae": 0.6},
            "local": {"macro_f1": 0.5, "mae": 0.9},
            "interval": {"macro_f1": 0.4, "mae": 1.2},
            "whole": {"macro_f1": 0.0, "mae": 3.0},
        }
        expected = sum(view_score(values[view]) for view in ("clean", "local", "interval")) / 3
        self.assertAlmostEqual(selection_score(values), expected)
        values["whole"] = {"macro_f1": 1.0, "mae": 0.0}
        self.assertAlmostEqual(selection_score(values), expected)

    def test_relative_degradation_sign(self):
        values = {
            "clean": {"macro_f1": 0.6, "mae": 0.6},
            "local": {"macro_f1": 0.5, "mae": 0.9},
            "interval": {"macro_f1": 0.4, "mae": 1.2},
        }
        changes = relative_degradation(values)
        self.assertAlmostEqual(changes["local"]["delta_macro_f1"], -0.1)
        self.assertAlmostEqual(changes["local"]["delta_mae"], 0.3)
        self.assertAlmostEqual(changes["interval"]["delta_macro_f1"], -0.2)
        self.assertAlmostEqual(changes["interval"]["delta_mae"], 0.6)


if __name__ == "__main__":
    unittest.main()
