from dataclasses import dataclass, field
from typing import List, Any

import numpy as np
from numpy.typing import NDArray

@dataclass
class GenerationParams:
    # Bates process parameters
    r: float  # Risk-free interest rate
    q: float  # Dividend rate
    v0: float  # Initial variance
    kappa: float  # Mean reversion speed
    theta: float  # Long-term variance
    sigma: float  # Volatility of variance
    rho: float  # Correlation between Brownian motions
    lambda_p: float  # Poisson jump rate
    spot_price: float  # Underlying asset spot price
    nu: float
    delta: float
    
    # Maturity and moneyness grids
    tau_grid: NDArray[np.uint16]  # List of time-to-maturities (in days)
    moneyness_grid: NDArray[np.float32]  # List of moneyness levels
