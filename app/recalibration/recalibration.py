from sklearn.preprocessing import MinMaxScaler
from app.data_gen.models import GenerationParams
from app.data_gen.surface_gen import generate_custom_130_point_iv_surface

import app.training.networks as net

from copy import deepcopy

import numpy as np
import QuantLib as ql
import torch

from numpy.typing import NDArray
from typing import  Tuple


########################### IVS GENERATION ##########################
def simulate_ivs_analyticaly(params: dict, moneyness_grid: NDArray, tau_grid: NDArray, spot_price: float, jump_means: NDArray, jump_variances: NDArray) -> Tuple[dict, GenerationParams]:
    generation_params = GenerationParams(
        moneyness_grid=moneyness_grid,
        tau_grid=tau_grid,
        spot_price=spot_price, 
        r=params['r'],
        q=params['q'],
        v0=params['v0'],
        kappa=params['kappa'],
        theta=params['theta'],
        sigma=params['sigma'],
        rho=params['rho'],
        lambda_p=params['lambda_p'],
        nu=jump_means,
        delta=jump_variances)
    
    data_point = generate_custom_130_point_iv_surface(params=generation_params)
    return data_point, generation_params


def simulate_ivs_with_neural_network(parameters: dict, model: net.NN1Residual, new_V: float | None, scaler: MinMaxScaler) -> torch.Tensor:
    parameters = deepcopy(parameters)
    if new_V is not None:
        parameters['v0'] = new_V
    raw_features = np.array(deepcopy(list(parameters.values()))).reshape((1, -1))
    norm_features = torch.from_numpy(scaler.transform(raw_features)).float()
    IVS_new = model(norm_features.to('cuda:0'))
    return IVS_new


################################# BATES STEP #####################################

def simulate_single_step_bates(params: GenerationParams) -> Tuple[float, float]:
    """
    Simulate a single time-step increment of the Bates process to update the state variables.

    The time increment (dt) is chosen as the smallest nonzero maturity in the tau_grid (converted
    from days to years). This single-step update provides the new state (asset price/return and 
    variance) that will be used to recalibrate the implied volatility surface (IVS), which is 
    spanned by all the maturities in the original tau_grid.

    Parameters:
        params (GenerationParams): An object containing the following attributes:
            - tau_grid: array-like; simulation time points (in days) for the process.
            - spot_price: float; the current asset price.
            - r: float; risk-free interest rate.
            - q: float; dividend yield.
            - v0: float; initial variance.
            - kappa: float; mean reversion speed of the variance.
            - theta: float; long-term mean variance.
            - sigma: float; volatility of volatility.
            - rho: float; correlation between asset and variance.
            - lambda_p: float; jump intensity.
            - nu: array-like; mean(s) of jump sizes (only the first element is used).
            - delta: array-like; standard deviation(s) of jump sizes (only the first element is used).

    Returns:
        Tuple[float, float]:
            - The updated asset price (or return) at time t = dt.
            - The updated variance at time t = dt.
    """
    # Determine the small time increment dt (in years) as the smallest nonzero maturity.
    dt = params.tau_grid[0] / 365.0  # assuming tau_grid is in days
    
    # Build the time grid for a single-step increment: [0, dt]
    time_grid = ql.TimeGrid([0.0, dt])
    
    # Set the evaluation date.
    today = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = today

    # Set up term structures for risk-free rate and dividend yield.
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(params.spot_price))
    risk_free_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(today, params.r, ql.Actual360())
    )
    dividend_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(today, params.q, ql.Actual360())
    )

    # Define the Bates Process.
    bates_process = ql.BatesProcess(
        risk_free_curve,    # Risk-free curve
        dividend_curve,     # Dividend yield curve
        spot_handle,        # Spot price
        params.v0,          # Initial variance
        params.kappa,       # Mean reversion speed
        params.theta,       # Long-term mean variance
        params.sigma,       # Volatility of volatility
        params.rho,         # Correlation between asset and variance
        params.lambda_p,    # Jump intensity
        params.nu[0],       # Mean of jump sizes
        params.delta[0]     # Standard deviation of jump sizes
    )

    # Get the number of state factors.
    dim = bates_process.factors()

    # Set up random number generators.
    uniform_rng = ql.UniformRandomSequenceGenerator(
        dim * (len(time_grid) - 1), ql.UniformRandomGenerator()
    )
    gaussian_rng = ql.GaussianRandomSequenceGenerator(uniform_rng)

    # Define the path generator for the single-step update.
    path_generator = ql.GaussianMultiPathGenerator(bates_process, time_grid, gaussian_rng, False)

    # Generate the sample path.
    sample_path = path_generator.next()
    multipath = sample_path.value()

    # Extract the asset price (or return) and variance at t = dt.
    # Note: Depending on the implementation, multipath[0] might be the asset price.
    updated_price = multipath[0][1]  # value at t = dt (second point)
    updated_variance = max(multipath[1][1], 0)  # enforce nonnegativity if needed

    # For debugging:
    print(f"Time grid: {[t for t in time_grid]}")
    # print(f"Asset prices along the path: {[multipath[0][i] for i in range(len(time_grid))]}")
    # print(f"Variances along the path: {[multipath[1][i] for i in range(len(time_grid))]}")
    print(f"Updated asset price at t = {dt}: {updated_price}")
    print(f"Updated variance at t = {dt}: {updated_variance}")

    return updated_price, updated_variance


################################# HESTON STEP #####################################

def update_parameters(parameters: dict, max_rel_change: float = 0.05) -> dict:
    """
    Update model parameters by adding a small Gaussian noise, ensuring that:
      - The relative change does not exceed max_rel_change (default: 5%).
      - The new values remain within their natural domains.
      - The Feller condition is always satisfied.

    Parameters:
        params (dict): Dictionary containing at least the following keys:
            - 'kappa': float, mean-reversion speed (assumed constant).
            - 'theta': float, long-term variance.
            - 'rho': float, correlation.
            - 'sigma': float, volatility-of-volatility.
        max_rel_change (float): Maximum allowed relative change (default is 0.05 for 5%).

    Returns:
        dict: A new dictionary with updated parameters.
              If the Feller condition (2*kappa*theta > sigma^2) is violated by the update,
              then 'sigma' and 'theta' are reverted to their previous values.
    """
    # Create a copy of the parameters to update.
    new_params = deepcopy(parameters)
    
    # Define the allowed domains for each parameter.
    domains = {
        'sigma': (0.01, 0.5),
        'theta': (0.01, 0.5),
        'rho': (-1.0, 1.0)
    }
    
    # For each parameter that is updated, add Gaussian noise
    for key in ['theta', 'sigma', 'rho']:
        current_val = parameters[key]
        # Define a standard deviation for the noise that is a small fraction of the current value.
        noise_std = 0.01 * abs(current_val)
        noise = np.random.normal(0, noise_std)
        
        # Check if the proposed change exceeds the maximum relative change.
        if abs(noise) > max_rel_change * abs(current_val):
            noise = np.sign(noise) * max_rel_change * abs(current_val)
        
        updated_val = current_val + noise
        
        # Clamp updated value to its allowed domain.
        lower_bound, upper_bound = domains[key]
        updated_val = np.clip(updated_val, lower_bound, upper_bound)
        
        new_params[key] = updated_val
    
    # Enforce the Feller condition for the variance process:
    # For the Heston model, the condition is 2*kappa*theta > sigma^2.
    sigma = parameters['sigma']
    if 2 * new_params['kappa'] * new_params['theta'] <= sigma ** 2:
        # If the Feller condition is violated, revert sigma and theta to their previous values.
        print("Feller condition violated; reverting 'sigma' and 'theta' to previous values.")
        new_params['sigma'] = parameters['sigma']
        new_params['theta'] = parameters['theta']
    
    return new_params


#################### GENERATION OF JUMP PARAMETERS ##################

def extract_jump_parameters_with_neural_networks(new_params: dict, IVS_new: NDArray,  model: net.NN2, scaler: MinMaxScaler) -> torch.Tensor:
    params = deepcopy(new_params)
    IVS = deepcopy(IVS_new)
    raw_features_2_part = np.array([params['theta'], params['sigma'], params['rho']]).reshape((1, -1))
    norm_features_2_part = scaler.transform(raw_features_2_part)
    norm_features_2 = torch.cat([
        torch.from_numpy(norm_features_2_part),
        torch.from_numpy(IVS)],
                                dim=1).float()
    model.eval()
    return model(norm_features_2.to("cuda:0"))[0]