import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
from efficient_frontier import (
    calculate_portfolio_metrics,
    maximize_return,
    maximize_sharpe_ratio
)

@st.cache_data(ttl=300)
def generate_efficient_frontier_with_tiered_allocation(returns, risk_free_rate=0.02, num_portfolios=30, 
                                                       aircraft_max_allocation=0.25, cash_min_allocation=0.05, cash_max_allocation=0.10, target_return=None):
    """
    Generate the efficient frontier with upper bound constraints on Aircraft allocation
    and cash (SHORT TERM) allocation.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame with returns for each asset
    risk_free_rate : float, optional
        Annual risk-free rate (default: 0.02 or 2%)
    num_portfolios : int, optional
        Number of portfolios to generate (default: 100)
    aircraft_max_allocation : float, optional
        Maximum allocation to Aircraft (default: 0.25 or 25%)
    cash_max_allocation : float, optional
        Maximum allocation to Cash/Short Term (default: 0.10 or 10%)
    target_return : float, optional
        Target return for constrained optimization (default: None)
    
    Returns:
    --------
    tuple
        (efficient_vols, efficient_returns, optimal_weights, optimal_return, optimal_vol)
    """
    # Get number of assets and asset names
    num_assets = len(returns.columns)
    asset_names = returns.columns
    
    # Find AIRCRAFT index if it exists
    aircraft_idx = None
    # Find SHORT TERM (cash) index if it exists
    cash_idx = None
    
    for i, col in enumerate(asset_names):
        if 'AIRCRAFT' in col:
            aircraft_idx = i
            print(f"Found AIRCRAFT at index {aircraft_idx}: {col}")
        elif 'SHORT TERM' in col:
            cash_idx = i
            print(f"Found SHORT TERM (cash) at index {cash_idx}: {col}")
    
    if aircraft_idx is None:
        print("No AIRCRAFT strategy found. Performing standard optimization.")
        # Import here to avoid circular import
        from efficient_frontier import generate_efficient_frontier
        return generate_efficient_frontier(returns, risk_free_rate, num_portfolios, target_return)
    
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12
    
    print(f"Aircraft maximum allocation constraint: {aircraft_max_allocation*100:.1f}%")
    
    # Define the bounds for each asset
    bounds = [(0, 1) for _ in range(num_assets)]
    
    # Special bound for Aircraft
    bounds[aircraft_idx] = (0, aircraft_max_allocation)
    
    # Special bound for Cash (SHORT TERM) - apply min and max constraints
    if cash_idx is not None:
        bounds[cash_idx] = (0, cash_max_allocation)  # Use parameter from UI
        print(f"Cash (SHORT TERM) maximum allocation constraint: {cash_max_allocation*100:.1f}%")
    
    # Define the constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
    ]
    
    # Add minimum constraint for Cash (SHORT TERM) from UI parameter
    if cash_idx is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, idx=cash_idx: x[idx] - cash_min_allocation  # Cash (SHORT TERM) allocation >= min value
        })
        print(f"Cash (SHORT TERM) minimum allocation constraint: {cash_min_allocation*100:.1f}%")
    
    # Find portfolio with maximum Sharpe ratio
    def neg_sharpe_ratio(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if portfolio_volatility == 0:
            return 0
        return -(portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Initial guess: equal weight portfolio
    init_guess = np.array([1.0/num_assets for _ in range(num_assets)])
    
    # Optimize for maximum Sharpe ratio
    max_sharpe_result = minimize(neg_sharpe_ratio, init_guess, method='SLSQP', 
                                bounds=bounds, constraints=constraints)
    max_sharpe_weights = max_sharpe_result['x']
    
    # Calculate portfolio metrics for max Sharpe ratio portfolio
    max_sharpe_return = np.sum(mean_returns * max_sharpe_weights)
    max_sharpe_vol = np.sqrt(np.dot(max_sharpe_weights.T, np.dot(cov_matrix, max_sharpe_weights)))
    
    print(f"Max Sharpe portfolio - Return: {max_sharpe_return*100:.2f}%, Vol: {max_sharpe_vol*100:.2f}%, Aircraft: {max_sharpe_weights[aircraft_idx]*100:.2f}%")
    
    # Generate efficient frontier
    target_vol_points = np.linspace(0.005, 0.3, num_portfolios)
    efficient_returns = []
    efficient_vols = []
    
    # For each target volatility, find the portfolio with maximum return
    for target_vol in target_vol_points:
        # Define the objective function: negative portfolio return
        def neg_portfolio_return(weights):
            return -np.sum(mean_returns * weights)
        
        # Define the volatility constraint
        def volatility_constraint(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_vol - target_vol
        
        # Add volatility constraint
        vol_constraint = {'type': 'eq', 'fun': volatility_constraint}
        all_constraints = constraints + [vol_constraint]
        
        try:
            # Optimize
            result = minimize(neg_portfolio_return, init_guess, method='SLSQP',
                            bounds=bounds, constraints=all_constraints)
            
            if result['success']:
                # Calculate portfolio metrics
                weights = result['x']
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                efficient_returns.append(portfolio_return)
                efficient_vols.append(portfolio_vol)
        except:
            # Skip this point if optimization fails
            pass
    
    # Calculate maximum achievable return portfolio instead of target return
    target_weights = None
    target_return_value = None
    target_vol = None
    
    # Always calculate the maximum achievable return portfolio
    print("Calculating maximum achievable return portfolio given constraints")
    # Define the objective function: maximize return (negative for minimization)
    def neg_portfolio_return(weights):
        return -np.sum(mean_returns * weights)
    
    try:
        # Optimize for maximum return
        print(f"Starting maximum return optimization with {len(returns.columns)} assets")
        
        # Try different initial guesses if the first one fails
        success = False
        for attempt in range(3):
            if attempt == 0:
                # Start with equal weights
                current_guess = init_guess
            elif attempt == 1:
                # Try max Sharpe weights as starting point
                current_guess = max_sharpe_weights
            else:
                # Try random weights that sum to 1
                random_weights = np.random.random(num_assets)
                current_guess = random_weights / np.sum(random_weights)
            
            target_result = minimize(neg_portfolio_return, current_guess, method='SLSQP',
                                    bounds=bounds, constraints=constraints)
            
            if target_result['success']:
                success = True
                break
            else:
                print(f"Max return optimization attempt {attempt+1} failed: {target_result['message']}")
        
        if success:
            target_weights = target_result['x']
            target_return_value = np.sum(mean_returns * target_weights)
            target_vol = np.sqrt(np.dot(target_weights.T, np.dot(cov_matrix, target_weights)))
            
            print(f"Maximum return portfolio optimization successful - Return: {target_return_value*100:.2f}%, Vol: {target_vol*100:.2f}%, Aircraft: {target_weights[aircraft_idx]*100:.2f}%")
        else:
            print(f"All maximum return optimization attempts failed")
            # Fall back to max Sharpe weights if optimization fails
            print(f"Falling back to max Sharpe weights for maximum return portfolio")
            target_weights = max_sharpe_weights.copy()
            target_return_value = np.sum(mean_returns * target_weights)
            target_vol = np.sqrt(np.dot(target_weights.T, np.dot(cov_matrix, target_weights)))
    except Exception as e:
        print(f"Error calculating maximum return weights: {str(e)}")
        # Fall back to max Sharpe weights if there's an error
        target_weights = max_sharpe_weights.copy() if max_sharpe_weights is not None else None
        if target_weights is not None:
            target_return_value = np.sum(mean_returns * target_weights)
            target_vol = np.sqrt(np.dot(target_weights.T, np.dot(cov_matrix, target_weights)))
            # Fall back to max Sharpe weights if target optimization fails
            print(f"Falling back to max Sharpe weights for target portfolio due to error")
            target_weights = max_sharpe_weights.copy()
            target_return_value = np.sum(mean_returns * target_weights)
            target_vol = np.sqrt(np.dot(target_weights.T, np.dot(cov_matrix, target_weights)))
    
    # For compatibility with the original function's return format
    return (efficient_vols, efficient_returns, max_sharpe_weights, max_sharpe_return, max_sharpe_vol, 
            target_weights, target_return_value, target_vol)
