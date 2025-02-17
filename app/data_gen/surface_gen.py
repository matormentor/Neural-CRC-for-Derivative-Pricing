from typing import Any, Dict
from logging import Logger

from QuantLib import Date, Settings, Actual365Fixed, FlatForward, YieldTermStructureHandle

from app.data_gen.models import GenerationParams
from app.data_gen.quantlib_setup.engine_setup import get_bates_engine
from app.data_gen.quantlib_setup.models import BatesProcessParams, OptionParams
from app.data_gen.quantlib_setup.option_setup import get_option_with_bates
from app.data_gen.sampling import *
from app.data_gen.implied_volatility import calculate_implied_volatility_approx

logger = Logger("dataGen")

# Generate implied volatility surface
def generate_130_point_iv_surface(spot_price, moneyness_grid, tau_grid, jumps_number=5):
    
    # Choose 5 maturities at random for when the jumps will happen respecting the piecewise constant function between adjacent maturities
    # NOTE: Inherent concentration of jumps on smaller maturities because of how we sampled the maturities in the first place
    selected_maturities = np.append(np.sort(rng.choice(tau_grid, size=jumps_number-1, replace=False)), 440) 
    
    jump_means, jump_variances = generate_means_and_volatilities(num_samples=jumps_number)
    
    interest_rate = sample_interest_rate()
    dividend_rate = sample_dividend_rate()
    mean_reversion_speed = sample_speed_mean_reversion()
    long_term_variance = sample_long_term_mean_of_variance()
    v0 = sample_initial_variance()
    volatility_of_variance = sample_inst_volatility_of_variance(
        mean_reversion_speed, long_term_variance
    )
    correlation = sample_motion_correlation()
    poisson_rate = sample_poisson_rate()
    
    fixed_params: Dict[str, Any] = {'r': interest_rate, 
                  'q': dividend_rate, 
                  'v0': v0, 
                  'kappa': mean_reversion_speed, 
                  'theta': long_term_variance,
                  'sigma': volatility_of_variance,
                  'rho': correlation,
                  'lambda_p': poisson_rate,
                  }
    
    data_point = fixed_params.copy()
    
    data_point.update({f"tau_{i}": tau for i, tau in enumerate(tau_grid, start=1)})
    data_point.update({f"m_{i}": m for i, m in enumerate(moneyness_grid, start=1)})
    data_point.update({f"nu_{i}": nu for i, nu in enumerate(jump_means, start=1)})
    data_point.update({f"delta_{i}": delta for i, delta in enumerate(jump_variances, start=1)})

    # QuantLib setup
    evaluation_date: Date = Date.todaysDate()
    Settings.instance().evaluationDate = evaluation_date
    
    # Yield curves
    day_count = Actual365Fixed()
    risk_free_curve = FlatForward(evaluation_date, interest_rate, day_count)
    dividend_curve = FlatForward(evaluation_date, dividend_rate, day_count)
    discount_curve = YieldTermStructureHandle(risk_free_curve)
    dividend_yield = YieldTermStructureHandle(dividend_curve)
    
    fixed_params.update({"spot_price": spot_price, 
                         'r': discount_curve, 
                         'q': dividend_yield})
        
    iv_surface = []
    
    for tau in tau_grid:
        nu, delta = get_jump_parameters(jump_means=jump_means, jump_variances=jump_variances, tau=tau, selected_maturities=selected_maturities)
        fixed_params.update({'nu': nu, 'delta': delta})
        bates_engine = get_bates_engine(BatesProcessParams(**fixed_params))
        for m in moneyness_grid:
            strike = spot_price / m
            
            option = get_option_with_bates(OptionParams(strike=strike, tau=tau, evaluation_date=evaluation_date, engine=bates_engine))
            
            market_price = option.NPV()
            forward_price = spot_price*np.exp((interest_rate - dividend_rate)* tau/365)
            implied_vol = calculate_implied_volatility_approx(
                Cm=market_price,
                K=strike,
                T=tau / 365,
                F=forward_price,
                r=interest_rate,
            )
            if np.isnan(implied_vol):
                raise Exception(f"Cm: {market_price}, K: {strike}, tai: {tau}, F: {forward_price}, r: {interest_rate}, implied_vol {implied_vol}")
            iv_surface.append(implied_vol)  

    data_point.update({'implied_vol_surface': iv_surface})
              
    return data_point


# Generate implied volatility surface
def generate_custom_130_point_iv_surface(params: GenerationParams):
    assert len(params.nu) == len(params.delta), "nu and delta must have the same length"
    jumps_number = len(params.nu)
    # Choose 5 maturities at random for when the jumps will happen respecting the piecewise constant function between adjacent maturities
    # NOTE: Inherent concentration of jumps on smaller maturities because of how we sampled the maturities in the first place
    selected_maturities = np.append(np.sort(rng.choice(params.tau_grid, size=jumps_number-1, replace=False)), 440) 
    
    jump_means, jump_variances = params.nu, params.delta
    
    interest_rate = params.r
    dividend_rate = params.q
    mean_reversion_speed = params.kappa
    long_term_variance = params.theta
    v0 = params.v0
    volatility_of_variance = params.sigma
    correlation = params.rho
    poisson_rate = params.lambda_p
    
    fixed_params: Dict[str, Any] = {'r': interest_rate, 
                  'q': dividend_rate, 
                  'v0': v0, 
                  'kappa': mean_reversion_speed, 
                  'theta': long_term_variance,
                  'sigma': volatility_of_variance,
                  'rho': correlation,
                  'lambda_p': poisson_rate,
                  }
    
    data_point = fixed_params.copy()
    
    data_point.update({f"tau_{i}": tau for i, tau in enumerate(params.tau_grid, start=1)})
    data_point.update({f"m_{i}": m for i, m in enumerate(params.moneyness_grid, start=1)})
    data_point.update({f"nu_{i}": nu for i, nu in enumerate(jump_means, start=1)})
    data_point.update({f"delta_{i}": delta for i, delta in enumerate(jump_variances, start=1)})

    
    
    # QuantLib setup
    evaluation_date: Date = Date.todaysDate()
    Settings.instance().evaluationDate = evaluation_date
    
    # Yield curves
    day_count = Actual365Fixed()
    risk_free_curve = FlatForward(evaluation_date, interest_rate, day_count)
    dividend_curve = FlatForward(evaluation_date, dividend_rate, day_count)
    discount_curve = YieldTermStructureHandle(risk_free_curve)
    dividend_yield = YieldTermStructureHandle(dividend_curve)
    
    fixed_params.update({"spot_price": params.spot_price, 
                         'r': discount_curve, 
                         'q': dividend_yield})
        
    iv_surface = []
    
    for tau in params.tau_grid:
        nu, delta = get_jump_parameters(jump_means=jump_means, jump_variances=jump_variances, tau=tau, selected_maturities=selected_maturities)
        fixed_params.update({'nu': nu, 'delta': delta})
        bates_engine = get_bates_engine(BatesProcessParams(**fixed_params))
        for m in params.moneyness_grid:
            strike = params.spot_price / m
            
            option = get_option_with_bates(OptionParams(strike=strike, tau=tau, evaluation_date=evaluation_date, engine=bates_engine))
            
            market_price = option.NPV()
            forward_price = params.spot_price*np.exp((interest_rate - dividend_rate)* tau/365.0)
            implied_vol = calculate_implied_volatility_approx(
                Cm=market_price,
                K=strike,
                T=tau / 365,
                F=forward_price,
                r=interest_rate,
            )
            if np.isnan(implied_vol):
                raise Exception("NAN")
            iv_surface.append(implied_vol)   

    data_point.update({'implied_vol_surface': iv_surface})
              
    return data_point


####################### UTILS ###########################

# Function to get nu and delta based on maturity
def get_jump_parameters(tau: int, jump_means: NDArray[np.float64], jump_variances: NDArray[np.float64], selected_maturities: NDArray[np.float64]):
    assert len(jump_means) == len(jump_variances), f"jump_mean and jump_variances do not have the same length"
    if len(jump_means) == 1:
        return jump_means[0], jump_variances[0]
    
    for i in range(len(selected_maturities)):
        if tau <= selected_maturities[i]:
            return jump_means[i], jump_variances[i]
    raise ValueError("Maturity out of range!")


# Generate implied volatility surface
def generate_130_point_prices(spot_price, moneyness_grid, tau_grid, jumps_number=1):
    
    # Choose 5 maturities at random for when the jumps will happen respecting the piecewise constant function between adjacent maturities
    # NOTE: Inherent concentration of jumps on smaller maturities because of how we sampled the maturities in the first place
    selected_maturities = np.append(np.sort(rng.choice(tau_grid, size=jumps_number-1, replace=False)), 440) 
    
    jump_means, jump_variances = generate_means_and_volatilities(num_samples=jumps_number)
    
    interest_rate = sample_interest_rate()
    dividend_rate = sample_dividend_rate()
    mean_reversion_speed = sample_speed_mean_reversion()
    long_term_variance = sample_long_term_mean_of_variance()
    v0 = sample_initial_variance()
    volatility_of_variance = sample_inst_volatility_of_variance(
        mean_reversion_speed, long_term_variance
    )
    correlation = sample_motion_correlation()
    poisson_rate = sample_poisson_rate()
    
    fixed_params: Dict[str, Any] = {'r': interest_rate, 
                  'q': dividend_rate, 
                  'v0': v0, 
                  'kappa': mean_reversion_speed, 
                  'theta': long_term_variance,
                  'sigma': volatility_of_variance,
                  'rho': correlation,
                  'lambda_p': poisson_rate,
                  }
    
    data_point = fixed_params.copy()
    
    data_point.update({f"tau_{i}": tau for i, tau in enumerate(tau_grid, start=1)})
    data_point.update({f"m_{i}": m for i, m in enumerate(moneyness_grid, start=1)})
    data_point.update({f"nu_{i}": nu for i, nu in enumerate(jump_means, start=1)})
    data_point.update({f"delta_{i}": delta for i, delta in enumerate(jump_variances, start=1)})

    
    
    # QuantLib setup
    evaluation_date: Date = Date.todaysDate()
    Settings.instance().evaluationDate = evaluation_date
    
    # Yield curves
    day_count = Actual365Fixed()
    risk_free_curve = FlatForward(evaluation_date, interest_rate, day_count)
    dividend_curve = FlatForward(evaluation_date, dividend_rate, day_count)
    discount_curve = YieldTermStructureHandle(risk_free_curve)
    dividend_yield = YieldTermStructureHandle(dividend_curve)
    
    fixed_params.update({"spot_price": spot_price, 
                         'r': discount_curve, 
                         'q': dividend_yield})
        
    prices = []
    iv_surface = []
    
    for tau in tau_grid:
        nu, delta = get_jump_parameters(jump_means=jump_means, jump_variances=jump_variances, tau=tau, selected_maturities=selected_maturities)
        fixed_params.update({'nu': nu, 'delta': delta})
        bates_engine = get_bates_engine(BatesProcessParams(**fixed_params))
        for m in moneyness_grid:
            strike = spot_price / m
            
            option = get_option_with_bates(OptionParams(strike=strike, tau=tau, evaluation_date=evaluation_date, engine=bates_engine))
            
            try:
                market_price = option.NPV()
                forward_price = spot_price*np.exp((interest_rate - dividend_rate)* tau/365)
                implied_vol = calculate_implied_volatility_approx(
                    Cm=market_price,
                    K=strike,
                    T=tau / 365,
                    F=forward_price,
                    r=interest_rate,
                )
                prices.append(market_price)
                iv_surface.append(implied_vol)

            except RuntimeError:
                prices.append(np.nan)  # Handle exceptions gracefully
                print("error")

    data_point.update({'prices': prices, 'iv_surface': iv_surface})
              
    return data_point
