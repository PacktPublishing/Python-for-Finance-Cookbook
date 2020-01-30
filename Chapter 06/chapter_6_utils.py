import numpy as np
from scipy.stats import norm


def simulate_gbm(s_0, mu, sigma, n_sims, T, N, random_seed=42, antithetic_var=False):
    '''
    Function used for simulating stock returns using Geometric Brownian Motion.
    
    Parameters
    ----------
    s_0 : float
        Initial stock price
    mu : float
        Drift coefficient
    sigma : float
        Diffusion coefficient
    n_sims : int
        Number of simulations paths
    dt : float
        Time increment, most commonly a day
    T : float
        Length of the forecast horizon, same unit as dt
    N : int
        Number of time increments in the forecast horizon
    random_seed : int
        Random seed for reproducibility
    antithetic_var : bool
        Boolean whether to use antithetic variates approach to reduce variance

    Returns
    -------
    S_t : np.ndarray
        Matrix (size: n_sims x (T+1)) containing the simulation results. 
        Rows respresent sample paths, while columns point of time.
    '''

    np.random.seed(random_seed)

    # time increment
    dt = T/N

    # Brownian
    if antithetic_var:
        dW_ant = np.random.normal(scale=np.sqrt(dt),
                                  size=(int(n_sims/2), N + 1))
        dW = np.concatenate((dW_ant, -dW_ant), axis=0)
    else:
        dW = np.random.normal(scale=np.sqrt(dt),
                              size=(n_sims, N + 1))

    # simulate the evolution of the process
    S_t = s_0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * dW,
                                 axis=1))
    S_t[:, 0] = s_0

    return S_t

def black_scholes_analytical(S_0, K, T, r, sigma, type='call'):
    '''
    Function used for calculating the price of European options using the analytical form of the Black-Scholes model.
    
    Parameters
    ------------
    s_0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Annualized risk-free rate
    sigma : float
        Standard deviation of the stock returns
    
    Returns
    -----------
    option_premium : float
        The premium on the option calculated using the Black-Scholes model
    '''

    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S_0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if type == 'call':
        option_premium = (S_0 * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1))
    elif type == 'put':
        option_premium = (K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S_0 * norm.cdf(-d1, 0, 1))
    else:
        raise ValueError('Wrong input for type!')

    return option_premium


def lsmc_american_option(S_0, K, T, N, r, sigma, n_sims, option_type, poly_degree, random_seed=42):
    '''
    Function used for calculating the price of American options using Least Squares Monte Carlo 
    algorithm of Longstaff and Schwartz (2001).
    
    Parameters
    ------------
    S_0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    N : int
        Number of time increments in the forecast horizon
    r : float
        Annualized risk-free rate
    sigma : float
        Standard deviation of the stock returns
    n_sims : int
        Number of paths to simulate
    option_type : str
        Type of the option. Allowable: ['call', 'put']
    poly_degree : int
        Degree of the polynomial to fit in the LSMC algorithm
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -----------
    option_premium : float
        The premium on the option 
    '''

    dt = T / N
    discount_factor = np.exp(-r * dt)

    gbm_simulations = simulate_gbm(s_0=S_0, mu=r, sigma=sigma, 
                                   n_sims=n_sims, T=T, N=N,
                                   random_seed=random_seed)

    if option_type == 'call':
        payoff_matrix = np.maximum(
            gbm_simulations - K, np.zeros_like(gbm_simulations))
    elif option_type == 'put':
        payoff_matrix = np.maximum(
            K - gbm_simulations, np.zeros_like(gbm_simulations))

    value_matrix = np.zeros_like(payoff_matrix)
    value_matrix[:, -1] = payoff_matrix[:, -1]

    for t in range(N - 1, 0, -1):
        regression = np.polyfit(
            gbm_simulations[:, t], value_matrix[:, t + 1] * discount_factor, poly_degree)
        continuation_value = np.polyval(regression, gbm_simulations[:, t])
        value_matrix[:, t] = np.where(payoff_matrix[:, t] > continuation_value,
                                      payoff_matrix[:, t],
                                      value_matrix[:, t + 1] * discount_factor)

    option_premium = np.mean(value_matrix[:, 1] * discount_factor)
    return option_premium
