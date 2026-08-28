"""
Unit tests for Probability Distributions, Tail Risk, and Expected Utility.
"""

import math
import unittest

from trading_bot.core.distributions import (
    GaussianDistribution, StudentTDistribution, SkewNormalDistribution,
    GaussianMixtureDistribution, EmpiricalSampleDistribution
)


class TestDistributions(unittest.TestCase):

    def test_gaussian_distribution(self):
        dist = GaussianDistribution(mu=0.002, sigma=0.02)
        self.assertAlmostEqual(dist.mean, 0.002)
        self.assertAlmostEqual(dist.variance, 0.0004)
        self.assertAlmostEqual(dist.std_dev, 0.02)

        # Symmetry around mean
        self.assertAlmostEqual(dist.cdf(0.002), 0.5)
        self.assertAlmostEqual(dist.inv_cdf(0.5), 0.002)

        # 95% VaR and CVaR
        var_95 = dist.var(alpha=0.05)
        cvar_95 = dist.cvar(alpha=0.05)
        self.assertGreater(cvar_95, var_95) # CVaR must be strictly worse than VaR

        # Sampling
        samples = dist.sample(500, seed=42)
        self.assertEqual(len(samples), 500)
        sample_mean = sum(samples) / len(samples)
        self.assertAlmostEqual(sample_mean, 0.002, delta=0.005)

    def test_student_t_distribution_fat_tails(self):
        # Student-t with df=4 has excess kurtosis = 6 / (4 - 4) = inf; df=5 -> excess kurtosis = 6
        t_dist = StudentTDistribution(df=5.0, mu=0.0, sigma=0.02)
        g_dist = GaussianDistribution(mu=0.0, sigma=t_dist.std_dev)

        self.assertAlmostEqual(t_dist.mean, 0.0)
        self.assertAlmostEqual(t_dist.excess_kurtosis, 6.0)

        # Student-t tail at 4 sigma should have higher density than Gaussian (heavy tail property)
        t_tail_pdf = t_dist.pdf(0.08)
        g_tail_pdf = g_dist.pdf(0.08)
        self.assertGreater(t_tail_pdf, g_tail_pdf)

        # Student-t CVaR should be higher than Gaussian
        t_cvar = t_dist.cvar(alpha=0.01)
        g_cvar = g_dist.cvar(alpha=0.01)
        self.assertGreater(t_cvar, g_cvar)

    def test_skew_normal_distribution(self):
        # Negative skewness (crypto/equity crash risk)
        dist = SkewNormalDistribution(mu=0.0, sigma=0.02, alpha=-3.0)
        self.assertLess(dist.skewness, 0.0)
        self.assertLess(dist.mean, 0.0)

    def test_gaussian_mixture_distribution(self):
        # 80% calm regime, 20% crash regime
        gmm = GaussianMixtureDistribution(
            weights=[0.8, 0.2],
            means=[0.001, -0.015],
            sigmas=[0.01, 0.05]
        )
        self.assertAlmostEqual(sum(gmm.weights), 1.0)
        expected_mu = 0.8 * 0.001 + 0.2 * (-0.015)
        self.assertAlmostEqual(gmm.mean, expected_mu)
        self.assertGreater(gmm.variance, 0.0)

    def test_empirical_sample_distribution(self):
        raw_data = [-0.04, -0.02, -0.01, 0.00, 0.005, 0.01, 0.02, 0.03, 0.06]
        emp = EmpiricalSampleDistribution(raw_data)
        self.assertAlmostEqual(emp.cdf(-0.04), 0.0, delta=0.15)
        self.assertAlmostEqual(emp.cdf(0.06), 1.0)
        self.assertAlmostEqual(emp.inv_cdf(0.5), 0.005, delta=0.01)


if __name__ == "__main__":
    unittest.main()

