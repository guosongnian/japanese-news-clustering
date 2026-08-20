"""不下载数据的快速单元测试。"""

import unittest

import numpy as np

from news_clustering import mean_pairwise_ari


class ClusteringSmokeTest(unittest.TestCase):
    def test_stability_is_permutation_invariant(self) -> None:
        first = np.array([0, 0, 1, 1])
        renamed = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(mean_pairwise_ari([first, renamed]), 1.0)


if __name__ == "__main__":
    unittest.main()
