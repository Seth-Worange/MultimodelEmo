"""验证缺失实验网格完整性。"""

import unittest

from scripts.robustness import build_cases


class RobustnessGridTest(unittest.TestCase):
    def test_all_combinations_locations_rates(self):
        cases = build_cases((0.1, 0.2, 0.4, 0.6, 0.8), ("start", "middle", "end", "random"))
        interval = [case for case in cases if case["pattern"] == "interval"]
        self.assertEqual(len(interval), 140)
        self.assertEqual(len({tuple(case.values()) for case in interval}), 140)
        self.assertEqual(len(cases), 165)


if __name__ == "__main__":
    unittest.main()
