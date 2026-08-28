"""
Mathematical and statistical utilities implemented in pure Python for high portability,
with zero external dependencies required.
"""

from __future__ import annotations
import math
import typing
from typing import Callable, List, Tuple, Optional, Sequence


# ==============================================================================
# Special Functions & Normal Distribution Approximations
# ==============================================================================

def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Standard or general normal probability density function."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (x - mu) / sigma
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * z * z)


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Standard or general normal cumulative distribution function using error function."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (x - mu) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_inv_cdf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Inverse normal CDF (quantile function / probit) using Peter John Acklam's
    rational approximation algorithm (relative error < 1.15e-9).
    """
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return float("-inf")
        if p == 1.0:
            return float("inf")
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Coefficients in rational approximations
    a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00
    ]
    b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        z = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        z = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        z = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    # One step of Halley's rational method for high precision refinement
    e = normal_cdf(z) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(0.5 * z * z)
    z = z - u / (1.0 + 0.5 * z * u)

    return mu + sigma * z


# ==============================================================================
# Gamma, Beta, and Student-t Functions
# ==============================================================================

def log_gamma(x: float) -> float:
    """Logarithm of the Gamma function via Lanczos approximation (g=7, N=9)."""
    if x <= 0:
        raise ValueError("x must be positive for log_gamma")
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.138571095836524,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]
    g = 7
    x -= 1.0
    a = p[0]
    t = x + g + 0.5
    for i in range(1, len(p)):
        a += p[i] / (x + i)
    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def gamma_fn(x: float) -> float:
    """Gamma function Gamma(x)."""
    return math.exp(log_gamma(x))


def beta_fn(a: float, b: float) -> float:
    """Beta function B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)."""
    return math.exp(log_gamma(a) + log_gamma(b) - log_gamma(a + b))


def betainc_continued_fraction(a: float, b: float, x: float, max_iter: int = 200, eps: float = 1e-12) -> float:
    """Continued fraction approximation for incomplete beta function (Lentz's method)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_h = d * c
        h *= del_h

        if abs(del_h - 1.0) <= eps:
            break

    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x < 0.0 or x > 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    bt = math.exp(log_gamma(a + b) - log_gamma(a) - log_gamma(b) + a * math.log(x) + b * math.log(1.0 - x))

    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betainc_continued_fraction(a, b, x) / a
    else:
        return 1.0 - bt * betainc_continued_fraction(b, a, 1.0 - x) / b


def student_t_pdf(x: float, df: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Student-t probability density function with location mu, scale sigma, and degrees of freedom df."""
    if df <= 0 or sigma <= 0:
        raise ValueError("df and sigma must be positive")
    z = (x - mu) / sigma
    coef = math.exp(log_gamma((df + 1.0) / 2.0) - log_gamma(df / 2.0)) / (math.sqrt(math.pi * df) * sigma)
    return coef * (1.0 + (z * z) / df) ** (-(df + 1.0) / 2.0)


def student_t_cdf(x: float, df: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Student-t cumulative distribution function."""
    if df <= 0 or sigma <= 0:
        raise ValueError("df and sigma must be positive")
    z = (x - mu) / sigma
    t2 = z * z
    x_beta = df / (df + t2)
    ib = regularized_incomplete_beta(df / 2.0, 0.5, x_beta)
    if z >= 0:
        return 1.0 - 0.5 * ib
    else:
        return 0.5 * ib


def student_t_inv_cdf(p: float, df: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Student-t quantile / inverse CDF via bisection & Newton refinement."""
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return float("-inf")
        if p == 1.0:
            return float("inf")
        raise ValueError(f"p must be in (0, 1), got {p}")

    z0 = normal_inv_cdf(p)
    if df > 2:
        factor = math.sqrt((df - 2.0) / df)
        x = z0 / factor
    else:
        x = z0

    low = -100.0 if p < 0.5 else 0.0
    high = 0.0 if p < 0.5 else 100.0

    while student_t_cdf(low, df) > p:
        low *= 2.0
    while student_t_cdf(high, df) < p:
        high *= 2.0

    for _ in range(60):
        mid = 0.5 * (low + high)
        cdf_mid = student_t_cdf(mid, df)
        if abs(cdf_mid - p) < 1e-10:
            x = mid
            break
        if cdf_mid < p:
            low = mid
        else:
            high = mid
        x = mid

    return mu + sigma * x


# ==============================================================================
# Numerical Integration & Quadrature
# ==============================================================================

_GL_NODES = [
    0.0,
    0.2011940939974345, -0.2011940939974345,
    0.3941513470775634, -0.3941513470775634,
    0.5709721726085388, -0.5709721726085388,
    0.7244177313601700, -0.7244177313601700,
    0.8482065834104272, -0.8482065834104272,
    0.9372985258406544, -0.9372985258406544,
    0.9879925180204854, -0.9879925180204854,
]
_GL_WEIGHTS = [
    0.2025782419255613,
    0.1984314853271116, 0.1984314853271116,
    0.1861610000155622, 0.1861610000155622,
    0.1662692058169939, 0.1662692058169939,
    0.1395706779261543, 0.1395706779261543,
    0.1071592204671719, 0.1071592204671719,
    0.0703660474881081, 0.0703660474881081,
    0.0307532419961173, 0.0307532419961173,
]


def integrate_quad(fn: Callable[[float], float], a: float, b: float, n_subintervals: int = 16) -> float:
    """Composite Gauss-Legendre quadrature for numerical integration from a to b."""
    if a == b:
        return 0.0
    if a > b:
        return -integrate_quad(fn, b, a, n_subintervals)

    h = (b - a) / n_subintervals
    total = 0.0

    for i in range(n_subintervals):
        sub_a = a + i * h
        sub_b = sub_a + h
        mid = 0.5 * (sub_a + sub_b)
        half_h = 0.5 * h

        sub_total = 0.0
        for node, weight in zip(_GL_NODES, _GL_WEIGHTS):
            x = mid + half_h * node
            sub_total += weight * fn(x)
        total += sub_total * half_h

    return total


# ==============================================================================
# 1D & Multidimensional Optimization Solvers
# ==============================================================================

def minimize_scalar_brent(
    fn: Callable[[float], float],
    bracket: Tuple[float, float],
    tol: float = 1e-6,
    max_iter: int = 100
) -> Tuple[float, float, int]:
    """Brent's method for 1D scalar minimization within a bracket (ax, bx)."""
    ax, cx = bracket
    bx = 0.5 * (ax + cx)
    golden_ratio = 0.3819660

    a = min(ax, cx)
    b = max(ax, cx)
    x = w = v = bx
    fx = fw = fv = fn(x)
    d = e = 0.0

    for iteration in range(1, max_iter + 1):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-10
        tol2 = 2.0 * tol1

        if abs(x - xm) <= (tol2 - 0.5 * (b - a)):
            return x, fx, iteration

        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            etemp = e
            e = d

            if abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (b - x):
                e = (a - x) if x >= xm else (b - x)
                d = golden_ratio * e
            else:
                d = p / q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if (xm - x) >= 0 else -tol1
        else:
            e = (a - x) if x >= xm else (b - x)
            d = golden_ratio * e

        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0 else -tol1))
        fu = fn(u)

        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu

    return x, fx, max_iter


def minimize_projected_gradient(
    fn: Callable[[List[float]], float],
    grad_fn: Optional[Callable[[List[float]], List[float]]],
    x0: List[float],
    bounds: Optional[List[Tuple[float, float]]] = None,
    learning_rate: float = 0.05,
    max_iter: int = 150,
    tol: float = 1e-6
) -> Tuple[List[float], float, int]:
    """Projected Gradient Descent with Backtracking Armijo Line Search."""
    n = len(x0)
    x = list(x0)
    
    if bounds:
        x = [max(bounds[i][0], min(bounds[i][1], x[i])) for i in range(n)]

    def compute_numerical_grad(curr_x: List[float]) -> List[float]:
        eps = 1e-7
        g = [0.0] * n
        base_f = fn(curr_x)
        for i in range(n):
            xp = list(curr_x)
            xp[i] += eps
            g[i] = (fn(xp) - base_f) / eps
        return g

    current_fx = fn(x)

    for iteration in range(1, max_iter + 1):
        grad = grad_fn(x) if grad_fn is not None else compute_numerical_grad(x)
        grad_norm = math.sqrt(sum(g * g for g in grad))
        if grad_norm < tol:
            return x, current_fx, iteration

        alpha = learning_rate
        c1 = 1e-4
        beta = 0.5

        best_cand = x
        best_cand_fx = current_fx
        found_step = False

        for _ in range(25):
            candidate = [x[i] - alpha * grad[i] for i in range(n)]
            if bounds:
                candidate = [max(bounds[i][0], min(bounds[i][1], candidate[i])) for i in range(n)]

            cand_fx = fn(candidate)
            step_norm_sq = sum((candidate[i] - x[i]) ** 2 for i in range(n))
            if cand_fx <= current_fx - c1 * step_norm_sq / (alpha + 1e-12):
                best_cand = candidate
                best_cand_fx = cand_fx
                found_step = True
                break
            alpha *= beta

        if not found_step:
            break

        step_dist = math.sqrt(sum((best_cand[i] - x[i]) ** 2 for i in range(n)))
        x = best_cand
        current_fx = best_cand_fx

        if step_dist < tol or abs(current_fx - best_cand_fx) < tol * 1e-2:
            return x, current_fx, iteration

    return x, current_fx, max_iter


# ==============================================================================
# Black-Scholes Analytical Pricing & Greeks Engine
# ==============================================================================

def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    is_call: bool = True
) -> float:
    """Computes analytical European Option price using the Black-Scholes-Merton model."""
    if spot <= 0 or strike <= 0 or volatility <= 0:
        return max(0.0, (spot - strike) if is_call else (strike - spot))
    if time_to_expiry <= 1e-8:
        return max(0.0, (spot - strike) if is_call else (strike - spot))

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    df_q = math.exp(-dividend_yield * time_to_expiry)
    df_r = math.exp(-rate * time_to_expiry)

    if is_call:
        price = spot * df_q * normal_cdf(d1) - strike * df_r * normal_cdf(d2)
    else:
        price = strike * df_r * normal_cdf(-d2) - spot * df_q * normal_cdf(-d1)

    return max(0.0, price)


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    is_call: bool = True
) -> dict[str, float]:
    """Calculates primary option Greeks: Delta, Gamma, Theta, Vega, Rho."""
    if spot <= 0 or strike <= 0 or volatility <= 0 or time_to_expiry <= 1e-8:
        intrinsic_delta = 1.0 if (is_call and spot > strike) else (-1.0 if (not is_call and spot < strike) else 0.0)
        return {"delta": intrinsic_delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    df_q = math.exp(-dividend_yield * time_to_expiry)
    df_r = math.exp(-rate * time_to_expiry)
    pdf_d1 = normal_pdf(d1)

    if is_call:
        delta = df_q * normal_cdf(d1)
    else:
        delta = -df_q * normal_cdf(-d1)

    gamma = (df_q * pdf_d1) / (spot * volatility * sqrt_t)
    vega = spot * df_q * sqrt_t * pdf_d1

    term1 = -(spot * df_q * pdf_d1 * volatility) / (2.0 * sqrt_t)
    if is_call:
        theta = term1 - rate * strike * df_r * normal_cdf(d2) + dividend_yield * spot * df_q * normal_cdf(d1)
    else:
        theta = term1 + rate * strike * df_r * normal_cdf(-d2) - dividend_yield * spot * df_q * normal_cdf(-d1)

    if is_call:
        rho = strike * time_to_expiry * df_r * normal_cdf(d2)
    else:
        rho = -strike * time_to_expiry * df_r * normal_cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }


def implied_volatility(
    target_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float = 0.0,
    is_call: bool = True,
    tol: float = 1e-5,
    max_iter: int = 50
) -> Optional[float]:
    """Inverts Black-Scholes equation to solve for Implied Volatility."""
    intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
    if target_price <= intrinsic + 1e-6:
        return 0.001

    vol = 0.30
    low_vol = 0.001
    high_vol = 5.0

    for _ in range(max_iter):
        p = black_scholes_price(spot, strike, time_to_expiry, rate, vol, dividend_yield, is_call)
        diff = p - target_price
        if abs(diff) < tol:
            return vol

        greeks = black_scholes_greeks(spot, strike, time_to_expiry, rate, vol, dividend_yield, is_call)
        vega = greeks["vega"]

        if vega > 1e-4:
            new_vol = vol - diff / vega
            if low_vol < new_vol < high_vol:
                vol = new_vol
                continue

        if diff > 0:
            high_vol = vol
        else:
            low_vol = vol
        vol = 0.5 * (low_vol + high_vol)

    return vol

