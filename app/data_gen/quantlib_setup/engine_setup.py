
from app.data_gen.quantlib_setup.models import BatesProcessParams

from QuantLib import BatesProcess, BatesModel, BatesEngine, QuoteHandle, SimpleQuote


def get_bates_engine(params: BatesProcessParams):
    spot_handle = QuoteHandle(SimpleQuote(params.spot_price))
    
    # Bates process
    bates_process = BatesProcess(
        riskFreeRate=params.r,
        dividendYield=params.q,
        s0=spot_handle,
        v0=params.v0,
        kappa=params.kappa,
        theta=params.theta,
        sigma=params.sigma,
        rho=params.rho,
        lambda_parameter=params.lambda_p,
        nu=params.nu,
        delta=params.delta,
    )

    # Set pricing engine
    bates_model = BatesModel(bates_process)
    return BatesEngine(bates_model)