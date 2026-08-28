"""
Probability Distribution representations for continuous financial return forecasts.
Supports analytical and numerical computation of moments, CDF, PDF, quantiles,
Expected Utility integration, Value-at-Risk (VaR), and Conditional Value-at-Risk (CVaR).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import math
import random
from typing import Callable, List, Optional, Sequence, Tuple

from trading_bot.core.math_utils import (
    normal_pdf, normal_cdf, normal_inv_cdf,
    student_t_pdf, student_t_cdf, student_t_inv_cdf,
    integrate_quad
)


class ProbDistribution(ABC):
    """
    Abstract Base Class for return probability distributions P(R_{t -> t+H} | F_t).
    """

    @property
    @abstractmethod
    def mean(self) -> float:
        """Expected return E[R]."""
        pass

    @property
    @abstractmethod
    def variance(self) -> float:
        """Variance Var(R)."""
        pass

    @property
    def std_dev(self) -> float:
        """Standard deviation sigma."""
        return math.sqrt(max(1e-12, self.variance))

    @property
    def skewness(self) -> float:
        """Third standardized moment."""
        return 0.0

    @property
    def excess_kurtosis(self) -> float:
        """Fourth standardized moment minus 3."""
        return 0.0

    @abstractmethod
    def pdf(self, x: float) -> float:
        """Probability density function f(x)."""
        pass

    @abstractmethod
    def cdf(self, x: float) -> float:
        """Cumulative distribution function F(x)."""
        pass

    @abstractmethod
    def inv_cdf(self, p: float) -> float:
        """Quantile function Q(p) = F^{-1}(p)."""
        pass

    @abstractmethod
    def sample(self, n: int, seed: Optional[int] = None) -> List[float]:
        """Generate n i.i.d random samples from this distribution."""
        pass

    def expected_utility(self, utility_fn: Callable[[float], float], weight: float, n_points: int = 32) -> float:
        """
        Calculates expected utility E[U(w * R)] = integral U(w * r) * f(r) dr.
        Integrates over a 6-sigma support window using composite Gauss-Legendre quadrature.
        """
        mu = self.mean
        sigma = self.std_dev
        lower = mu - 6.0 * sigma
        upper = mu + 6.0 * sigma

        def integrand(r: float) -> float:
            p_val = self.pdf(r)
            if p_val <= 0:
                return 0.0
            return utility_fn(weight * r) * p_val

        return integrate_quad(integrand, lower, upper, n_subintervals=n_points)

    def var(self, alpha: float = 0.05) -> float:
        """
        Value-at-Risk at confidence level alpha (e.g. 5% worst loss).
        Returns positive value representing maximum expected loss at alpha percentile.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        # Quantile of return
        q = self.inv_cdf(alpha)
        return -q

    def cvar(self, alpha: float = 0.05, n_steps: int = 20) -> float:
        """
        Conditional Value-at-Risk (Expected Shortfall) at confidence alpha.
        CVaR_alpha = - (1 / alpha) * integral_0^alpha Q(p) dp.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")

        def integrand(p: float) -> float:
            return self.inv_cdf(p)

        integrated_tail = integrate_quad(integrand, 1e-6, alpha, n_subintervals=n_steps)
        return -(1.0 / alpha) * integrated_tail


class GaussianDistribution(ProbDistribution):
    """
    Parametric Normal Distribution N(mu, sigma^2).
    """

    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self._mu = float(mu)
        self._sigma = float(sigma)

    @property
    def mean(self) -> float:
        return self._mu

    @property
    def variance(self) -> float:
        return self._sigma * self._sigma

    @property
    def skewness(self) -> float:
        return 0.0

    @property
    def excess_kurtosis(self) -> float:
        return 0.0

    def pdf(self, x: float) -> float:
        return normal_pdf(x, self._mu, self._sigma)

    def cdf(self, x: float) -> float:
        return normal_cdf(x, self._mu, self._sigma)

    def inv_cdf(self, p: float) -> float:
        return normal_inv_cdf(p, self._mu, self._sigma)

    def sample(self, n: int, seed: Optional[int] = None) -> List[float]:
        rng = random.Random(seed) if seed is not None else random
        return [rng.gauss(self._mu, self._sigma) for _ in range(n)]

    def var(self, alpha: float = 0.05) -> float:
        z_alpha = normal_inv_cdf(alpha)
        return -(self._mu + self._sigma * z_alpha)

    def cvar(self, alpha: float = 0.05, n_steps: int = 20) -> float:
        # Exact closed-form for normal distribution
        z_alpha = normal_inv_cdf(alpha)
        return -self._mu + self._sigma * (normal_pdf(z_alpha) / alpha)

    def __repr__(self) -> str:
        return f"Gaussian(mu={self._mu:.5f}, sigma={self._sigma:.5f})"


class StudentTDistribution(ProbDistribution):
    """
    Student-t Distribution t_nu(mu, sigma) with fat tails.
    Crucial for financial asset modeling where kurtosis > 3.
    """

    def __init__(self, df: float, mu: float = 0.0, sigma: float = 1.0):
        if df <= 2.0:
            raise ValueError("df must be > 2 for finite variance")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self._df = float(df)
        self._mu = float(mu)
        self._sigma = float(sigma)

    @property
    def degrees_of_freedom(self) -> float:
        return self._df

    @property
    def mean(self) -> float:
        return self._mu

    @property
    def variance(self) -> float:
        return self._sigma * self._sigma * (self._df / (self._df - 2.0))

    @property
    def skewness(self) -> float:
        return 0.0

    @property
    def excess_kurtosis(self) -> float:
        if self._df > 4.0:
            return 6.0 / (self._df - 4.0)
        return float("inf")

    def pdf(self, x: float) -> float:
        return student_t_pdf(x, self._df, self._mu, self._sigma)

    def cdf(self, x: float) -> float:
        return student_t_cdf(x, self._df, self._mu, self._sigma)

    def inv_cdf(self, p: float) -> float:
        return student_t_inv_cdf(p, self._df, self._mu, self._sigma)

    def sample(self, n: int, seed: Optional[int] = None) -> List[float]:
        rng = random.Random(seed) if seed is not None else random
        samples = []
        for _ in range(n):
            # t_nu = Z / sqrt(V / nu) where Z ~ N(0,1), V ~ ChiSq(nu) = Gamma(nu/2, 2)
            z = rng.gauss(0.0, 1.0)
            v = rng.gammavariate(self._df / 2.0, 2.0)
            t_val = z / math.sqrt(v / self._df)
            samples.append(self._mu + self._sigma * t_val)
        return samples

    def __repr__(self) -> str:
        return f"StudentT(df={self._df:.2f}, mu={self._mu:.5f}, sigma={self._sigma:.5f})"


class SkewNormalDistribution(ProbDistribution):
    """
    Azzalini's Skew-Normal Distribution SN(mu, sigma, alpha).
    Captures directional asymmetry (negative skewness in equity/crypto crashes).
    """

    def __init__(self, mu: float, sigma: float, alpha: float = 0.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self._mu = float(mu)
        self._sigma = float(sigma)
        self._alpha = float(alpha)
        # Delta factor
        self._delta = self._alpha / math.sqrt(1.0 + self._alpha * self._alpha)

    @property
    def mean(self) -> float:
        return self._mu + self._sigma * self._delta * math.sqrt(2.0 / math.pi)

    @property
    def variance(self) -> float:
        return self._sigma * self._sigma * (1.0 - 2.0 * (self._delta ** 2) / math.pi)

    @property
    def skewness(self) -> float:
        delta_sq = self._delta ** 2
        numerator = (4.0 - math.pi) / 2.0 * (self._delta * math.sqrt(2.0 / math.pi)) ** 3
        denominator = (1.0 - 2.0 * delta_sq / math.pi) ** 1.5
        return numerator / denominator

    def pdf(self, x: float) -> float:
        z = (x - self._mu) / self._sigma
        return (2.0 / self._sigma) * normal_pdf(z) * normal_cdf(self._alpha * z)

    def cdf(self, x: float) -> float:
        # Numerical integration for skew normal CDF
        lower = self.mean - 8.0 * self.std_dev
        if x <= lower:
            return 0.0
        return max(0.0, min(1.0, integrate_quad(self.pdf, lower, x, n_subintervals=16)))

    def inv_cdf(self, p: float) -> float:
        if not (0.0 < p < 1.0):
            if p == 0.0:
                return float("-inf")
            if p == 1.0:
                return float("inf")
            raise ValueError(f"p must be in (0, 1), got {p}")

        low = self.mean - 8.0 * self.std_dev
        high = self.mean + 8.0 * self.std_dev

        for _ in range(50):
            mid = 0.5 * (low + high)
            c = self.cdf(mid)
            if abs(c - p) < 1e-7:
                return mid
            if c < p:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    def sample(self, n: int, seed: Optional[int] = None) -> List[float]:
        rng = random.Random(seed) if seed is not None else random
        samples = []
        d = self._delta
        for _ in range(n):
            u0 = rng.gauss(0.0, 1.0)
            u1 = rng.gauss(0.0, 1.0)
            if u0 >= 0:
                z = d * u0 + math.sqrt(1.0 - d * d) * u1
            else:
                z = -d * u0 + math.sqrt(1.0 - d * d) * u1
            samples.append(self._mu + self._sigma * z)
        return samples

    def __repr__(self) -> str:
        return f"SkewNormal(mu={self._mu:.5f}, sigma={self._sigma:.5f}, alpha={self._alpha:.2f})"


class GaussianMixtureDistribution(ProbDistribution):
    """
    Gaussian Mixture Model sum_{k=1}^K w_k N(mu_k, sigma_k^2) for multimodal regime switching.
    """

    def __init__(self, weights: Sequence[float], means: Sequence[float], sigmas: Sequence[float]):
        if not (len(weights) == len(means) == len(sigmas)):
            raise ValueError("weights, means, and sigmas must have the same length")
        total_w = sum(weights)
        if total_w <= 0:
            raise ValueError("weights must sum to a positive number")
        self._weights = [w / total_w for w in weights]
        self._means = [float(m) for m in means]
        self._sigmas = [float(s) for s in sigmas]
        for s in self._sigmas:
            if s <= 0:
                raise ValueError("all sigmas must be positive")

    @property
    def weights(self) -> List[float]:
        return list(self._weights)

    @property
    def means(self) -> List[float]:
        return list(self._means)

    @property
    def sigmas(self) -> List[float]:
        return list(self._sigmas)

    @property
    def mean(self) -> float:
        return sum(w * m for w, m in zip(self._weights, self._means))

    @property
    def variance(self) -> float:
        mu = self.mean
        return sum(w * (s * s + (m - mu) ** 2) for w, m, s in zip(self._weights, self._means, self._sigmas))

    def pdf(self, x: float) -> float:
        return sum(w * normal_pdf(x, m, s) for w, m, s in zip(self._weights, self._means, self._sigmas))

    def cdf(self, x: float) -> float:
        return sum(w * normal_cdf(x, m, s) for w, m, s in zip(self._weights, self._means, self._sigmas))

    def inv_cdf(self, p: float) -> float:
        if not (0.0 < p < 1.0):
            if p == 0.0:
                return float("-inf")
            if p == 1.0:
                return float("inf")
            raise ValueError(f"p must be in (0, 1), got {p}")

        low = self.mean - 8.0 * self.std_dev
        high = self.mean + 8.0 * self.std_dev

        for _ in range(50):
            mid = 0.5 * (low + high)
            c = self.cdf(mid)
            if abs(c - p) < 1e-7:
                return mid
            if c < p:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    def sample(self, n: int, seed: Optional[int] = None) -> List[float]:
        rng = random.Random(seed) if seed is not None else random
        samples = []
        cum_weights = []
        acc = 0.0
        for w in self._weights:
            acc += w
            cum_weights.append(acc)

        for _ in range(n):
            r = rng.random()
            idx = 0
            for i, cw in enumerate(cum_weights):
                if r <= cw:
                    idx = i
                    break
            samples.append(rng.gauss(self._means[idx], self._sigmas[idx]))
        return samples

    def __repr__(self) -> str:
        comps = [f"{w:.2f}*N({m:.4f},{s:.4f})" for w, m, s in zip(self._weights, self._means, self._sigmas)]
        return f"GMM({', '.join(comps)})"


class EmpiricalSampleDistribution(ProbDistribution):
    """
    Non-parametric empirical distribution constructed from historical or simulated Monte Carlo returns.
    """

    def __init__(self, samples: Sequence[float]):
        if len(samples) < 2:
            raise ValueError("At least 2 samples required")
        self._samples = sorted(float(x) for x in samples)
        self._n = len(self._samples)
        self._mean = sum(self._samples) / self._n
        self._var = sum((x - self._mean) ** 2 for x in self._samples) / (self._n - 1)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._var

    def pdf(self, x: float) -> float:
        # Gaussian Kernel Density Estimate with Silverman's rule of thumb bandwidth
        sigma = self.std_dev
        h = 1.06 * sigma * (self._n ** (-0.2))
        h = max(h, 1e-6)
        return (1.0 / (self._n * h)) * sum(normal_pdf((x - xi) / h) for xi in self._samples)

    def cdf(self, x: float) -> float:
        if x <= self._samples[0]:
            return 0.0
        if x >= self._samples[-1]:
            return 1.0
        # Linear interpolation of empirical CDF
        import bisect
        idx = bisect.bisect_right(self._samples, x)
        return idx / self._n

    def inv_cdf(self, p: float) -> float:
        if not (0.0 < p < 1.0):
            if p == 0.0:
                return self._samples[0]
            if p == 1.0:
                return self._samples[-1]
            raise ValueError(f"p must be in (0, 1), got {p}")

        target_idx = p * (self._n - 1)
        low_idx = int(target_idx)
        high_idx = min(self._n - 1, low_idx + 1)
        weight = target_idx - low_idx
        return (1.0 - weight) * self._samples[low_idx] + weight * self._samples[high_idx]

    def sample(self, n: int, seed: Optional[int] = None) -> List[float]:
        rng = random.Random(seed) if seed is not None else random
        return [rng.choice(self._samples) for _ in range(n)]

    def __repr__(self) -> str:
        return f"Empirical(N={self._n}, mean={self._mean:.5f}, std={self.std_dev:.5f})"

