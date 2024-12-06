from typing import Tuple

import numpy as np
from numpy.typing import NDArray


# Seed and create the random generator
rng = np.random.default_rng()

# tau
def sample_maturities(num_samples: int = 10) -> NDArray[np.uint16]:
    tau_min, tau_max = 7, 440
    return np.logspace(np.log10(tau_min), np.log10(tau_max), num=num_samples, dtype=np.uint16)

# m
def sample_moneyness(num_samples: int = 13) -> NDArray[np.float64]:
    m_min, m_max = 0.8, 1.2
    return np.linspace(m_min, m_max, num=num_samples, endpoint=True, dtype=np.float64)
    

# r
def sample_interest_rate(rng: np.random.Generator = rng) -> float:
    return rng.uniform(0.005, 0.06 + np.finfo(float).eps)

# q
def sample_dividend_rate(rng: np.random.Generator = rng) -> float:
    return rng.uniform(0.002, 0.03 + np.finfo(float).eps)

# k
def sample_speed_mean_reversion(rng: np.random.Generator = rng) -> float:
    return rng.uniform(5, 10 + np.finfo(float).eps)


# Hull White Bates model

# Theta
def sample_long_term_mean_of_variance(rng: np.random.Generator = rng) -> float:
    return rng.uniform(0.1, 0.4 + np.finfo(float).eps)

# Sigma
def sample_inst_volatility_of_variance(k: float, theta: float, rng: np.random.Generator = rng) -> float:
    return rng.uniform(0.2, np.sqrt(2*k*theta))  # Feller Condition

# Rho
def sample_motion_correlation(rng: np.random.Generator = rng) -> float:
    return rng.uniform(-0.3, 0.3 + np.finfo(float).eps)

# Lambda 
def sample_poisson_rate(rng: np.random.Generator = rng) -> float:
    return rng.uniform(0.05, 0.5 + np.finfo(float).eps)


def generate_means_and_volatilities(rng: np.random.Generator = rng, num_samples: int = 5) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Parameters are in the range of plausible values
    means = rng.uniform(-0.3, 0.3 + np.finfo(float).eps, num_samples)  # Jump means
    volatilities = rng.uniform(0.2, 0.3 + np.finfo(float).eps, num_samples)  # Jump volatilites
    
    return means, volatilities

# Initial Variance V0
def sample_initial_variance(rng: np.random.Generator = rng) -> float:
    return rng.uniform(0.0001, 0.11 + np.finfo(float).eps)
