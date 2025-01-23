import numpy as np
eps = np.finfo(float).eps

def A_f(x: float) -> float:
    """
    Computes the A(x) function based on Poyla 1949.

    Parameters:
    x : float or numpy array
        Input value(s) for which A(x) is calculated.

    Returns:
    float or numpy array
        The value of A(x) for the input.
    """
    term = np.sqrt(1 - np.exp(-2 * x**2 / np.pi))
    result = 0.5 + 0.5 * np.sign(x) * term
    return result


def calculate_implied_volatility_approx(Cm: float, K: float, T: float, F: float, r: float) -> float:
    """
    Calculate implied volatility approximation using the formula in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2908494.
    
    Parameters
    ----------
        Cm : float
            Market price of the call option
        K  : float
            Option strike
        T  : float
            Maturity
        F  : float
            Forward price at T (maturity) of the underlying (stock)
        r  : float
            (constant) Interest rate
    
    Returns
    -------
        float 
            The implied volatility approximation
    """
    
    y = np.log(F / K)
    
    if Cm <= 0:
        alpha_c = max(np.expm1(y), 0) + eps  # Ensure the fact that alpha_c > max(e^y - 1, 0) added epsilon for numerical stability
    else:
        alpha_c = Cm/(K * np.exp(-r * T)) 
        
    R = 2 * alpha_c - np.exp(y) + 1
    
    # A, B, C computations
    A = (np.exp((1 - (2 / np.pi)) * y) - np.exp(-(1 - (2 / np.pi)) * y)) ** 2
    B = (4 * (np.exp((2 / np.pi) * y) + np.exp((-2 / np.pi) * y)) -
        (2 * np.exp(-y) * (np.exp((1 - 2 / np.pi) * y) + np.exp(-(1 - 2 / np.pi) * y)) * (np.exp(2 * y) + 1 - R**2)))
    C = max(np.exp(-2 * y) * (R**2 - (np.expm1(y))**2) * ((np.exp(y) + 1)**2 - R**2), eps)
    
    # More helper variables
    beta = 2 * C / (B + np.sqrt(B**2 + 4 * A * C))
    gamma = -(np.pi/2) * np.log(beta) 

    # Due to numerical Stability
    d1 = max(gamma + y, 0)
    d2 = max(gamma - y, 0)
     
    # Determine implied volatility approximation
    if y >= 0:
        C_0 = K * np.exp(-r * T) * (np.exp(y) * A_f(np.sqrt(2 * y)) - 0.5)
        if Cm <= C_0:
            sigma = (1 / np.sqrt(T)) * (np.sqrt(d1) - np.sqrt(d2))
        else:
            sigma = (1 / np.sqrt(T)) * (np.sqrt(d1) + np.sqrt(d2))
    else:
        C_0 = K * np.exp(-r * T) * (0.5 * np.exp(y) - A_f(-np.sqrt(-2 * y)))
        if Cm <= C_0:
            sigma = (1 / np.sqrt(T)) * (-np.sqrt(d1) + np.sqrt(d2))
        else:
            sigma = (1 / np.sqrt(T)) * (np.sqrt(d1) + np.sqrt(d2))
            
    if np.isnan(sigma):
        raise Exception(f"Cm: {Cm}, K: {K}, tai: {T}, F: {F}, r: {r} \n alpha_c: {alpha_c}, y: {y}, R: {R}, A: {A}, B: {B}, C: {C}, beta: {beta}, gamma: {gamma}, C_0: {C_0}, sigma: {sigma}")
            
    return sigma
