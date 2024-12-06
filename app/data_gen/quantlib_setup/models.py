from dataclasses import dataclass
from QuantLib import YieldTermStructureHandle, Date, BatesEngine # type: ignore

@dataclass
class BatesProcessParams:
    spot_price: float
    r: YieldTermStructureHandle
    q: YieldTermStructureHandle
    v0: float  # Initial variance
    kappa: float  # Mean reversion speed
    theta: float  # Long-term variance
    sigma: float  # Volatility of variance
    rho: float  # Correlation between asset and variance
    lambda_p: float  # Poisson jump intensity
    nu: float  # Mean jump size
    delta: float  # Jump size volatility
    
    
@dataclass
class OptionParams:
    strike: float
    tau: int
    evaluation_date: Date
    engine: BatesEngine
    