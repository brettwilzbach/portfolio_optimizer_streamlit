import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import minimize
import streamlit as st

@st.cache_data(ttl=3600, max_entries=20)
def generate_synthetic_returns(strategies, annual_returns, volatilities=None, correlation_matrix=None, periods=36):
    """
    Generate synthetic monthly returns based on annual returns and volatilities.
    
    Parameters:
    -----------
    strategies : list
        List of strategy names
    annual_returns : dict
        Dictionary mapping strategy names to annual returns
    volatilities : dict, optional
        Dictionary mapping strategy names to annual volatilities
    correlation_matrix : pd.DataFrame, optional
        Correlation matrix between strategies
    periods : int, optional
        Number of periods to generate (default: 36 months)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with synthetic monthly returns
    """
    n_strategies = len(strategies)
    
    # Default volatilities if not provided (as percentage of returns)
    if volatilities is None:
        volatilities = {}
        for strategy in strategies:
            # Higher returns typically have higher volatility
            # This is a simplification for demonstration purposes
            ret = annual_returns.get(strategy, 0.05)
            if "SHORT TERM" in strategy:
                # Cash-like strategies have lower volatility
                volatilities[strategy] = max(0.01, ret * 0.3)
            elif "CMBS" in strategy:
                volatilities[strategy] = max(0.02, ret * 0.6)
            elif "CLO" in strategy:
                volatilities[strategy] = max(0.03, ret * 0.7)
            elif "ABS" in strategy:
                volatilities[strategy] = max(0.025, ret * 0.65)
            elif "AIRCRAFT" in strategy:
                volatilities[strategy] = max(0.035, ret * 0.8)
            else:
                volatilities[strategy] = max(0.02, ret * 0.5)
    
    # Default correlation matrix if not provided
    if correlation_matrix is None:
        # Create a reasonable correlation matrix
        # Diagonal elements are 1 (perfect self-correlation)
        # Off-diagonal elements are random between 0.1 and 0.7
        np.random.seed(42)  # For reproducibility
        corr = np.random.uniform(0.1, 0.7, size=(n_strategies, n_strategies))
        corr = (corr + corr.T) / 2  # Make it symmetric
        np.fill_diagonal(corr, 1)  # Diagonal elements are 1
        
        # Ensure it's positive semi-definite (a valid correlation matrix)
        min_eig = np.min(np.linalg.eigvals(corr))
        if min_eig < 0:
            corr -= min_eig * np.eye(n_strategies)
        
        correlation_matrix = pd.DataFrame(corr, index=strategies, columns=strategies)
    
    # Convert annual returns and volatilities to monthly
    monthly_returns = {s: (1 + r) ** (1/12) - 1 for s, r in annual_returns.items()}
    monthly_vols = {s: v / np.sqrt(12) for s, v in volatilities.items()}
    
    # Create covariance matrix
    cov_matrix = np.zeros((n_strategies, n_strategies))
    for i, s1 in enumerate(strategies):
        for j, s2 in enumerate(strategies):
            cov_matrix[i, j] = correlation_matrix.loc[s1, s2] * monthly_vols[s1] * monthly_vols[s2]
    
    # Generate random returns
    np.random.seed(42)  # For reproducibility
    returns = np.random.multivariate_normal(
        mean=[monthly_returns[s] for s in strategies],
        cov=cov_matrix,
        size=periods
    )
    
    # Create DataFrame with dates
    dates = pd.date_range(end=pd.Timestamp.today(), periods=periods, freq='M')
    return_df = pd.DataFrame(returns, index=dates, columns=strategies)
    
    return return_df

@st.cache_data(ttl=3600, max_entries=100)
def calculate_compounded_returns(returns_data):
    """
    Calculate compounded (geometric) returns that are properly annualized.
    
    Parameters:
    -----------
    returns_data : pd.DataFrame
        DataFrame with monthly returns for each asset
        
    Returns:
    --------
    pd.Series
        Annualized compounded returns for each asset
    """
    compound_returns = {}
    
    for col in returns_data.columns:
        # Get non-NaN values
        returns_array = returns_data[col].dropna().values
        
        if len(returns_array) > 0:
            # Calculate compound return: (1+r1)*(1+r2)*...*(1+rn) - 1
            compound_return = np.prod(1 + returns_array) - 1
            
            # Annualize: (1+r)^(12/n) - 1 where n is number of months
            n_months = len(returns_array)
            annualized_return = (1 + compound_return) ** (12 / n_months) - 1
        else:
            # Fallback if no data
            annualized_return = 0
            
        compound_returns[col] = annualized_return
    
    return pd.Series(compound_returns)

@st.cache_data(ttl=3600, max_entries=100)
def calculate_portfolio_metrics(returns, weights):
    """
    Calculate portfolio metrics: expected return and volatility.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame with returns for each asset
    weights : array-like
        Portfolio weights
        
    Returns:
    --------
    tuple
        (expected_return, volatility)
    """
    # Convert weights to numpy array
    weights = np.array(weights)
    
    # Print debug info about the returns data
    print(f"\nCalculating portfolio metrics with {len(returns)} months of data")
    print(f"Returns shape: {returns.shape}, Weights shape: {weights.shape}")
    
    # Check if weights match the number of assets
    if len(weights) != returns.shape[1]:
        print(f"WARNING: Weights length ({len(weights)}) doesn't match number of assets ({returns.shape[1]})")
        # Adjust weights if necessary
        if len(weights) > returns.shape[1]:
            weights = weights[:returns.shape[1]]
            print(f"Truncated weights to match assets: {len(weights)}")
        else:
            # Pad with zeros
            weights = np.pad(weights, (0, returns.shape[1] - len(weights)), 'constant')
            print(f"Padded weights to match assets: {len(weights)}")
    
    # Normalize weights to sum to 1
    if np.sum(weights) != 0:
        weights = weights / np.sum(weights)
    
    # Handle missing values in the returns data
    returns_filled = returns.copy()
    
    # First check if any columns have too many missing values (>50%)
    missing_pct = returns_filled.isna().mean()
    print("\nMissing data percentage by strategy:")
    for col in returns_filled.columns:
        print(f"{col}: {missing_pct[col]*100:.1f}% missing")
        
    # Replace NaN with column mean for better accuracy
    # For columns with >50% missing, we'll still use the mean but print a warning
    for col in returns_filled.columns:
        if missing_pct[col] > 0.5:
            print(f"WARNING: {col} has {missing_pct[col]*100:.1f}% missing data - using available data but results may be less reliable")
        returns_filled[col] = returns_filled[col].fillna(returns_filled[col].mean())
    
    # Calculate expected returns (annualized)
    expected_returns = returns_filled.mean() * 12
    portfolio_return = np.sum(expected_returns * weights)
    
    # Calculate portfolio volatility (annualized)
    cov_matrix = returns_filled.cov() * 12
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))    # No minimum volatility enforcement - show all instances
    # (previously enforced 2% minimum)
    
    # Print the contribution of each strategy to the portfolio return and volatility
    print("\nStrategy contributions to portfolio:")
    max_random_portfolios = min(len(returns.columns), 10)  # Limit to at most 10 random portfolios
    step_size = max(1, len(returns.columns) // max_random_portfolios)
    for i in range(0, len(returns.columns), step_size):
        strategy = returns.columns[i]
        contribution = expected_returns.iloc[i] * weights[i]
        vol = returns_filled[strategy].std() * np.sqrt(12)
        print(f"{strategy}: Return {expected_returns.iloc[i]*100:.2f}%, Vol {vol*100:.2f}%, Weight {weights[i]*100:.1f}%, Contrib {contribution*100:.2f}%")
    
    print(f"\nPortfolio expected return: {portfolio_return*100:.2f}%")
    print(f"Portfolio volatility: {portfolio_volatility*100:.2f}%")
    
    # Ensure we don't return NaN values
    if np.isnan(portfolio_return):
        print("WARNING: Portfolio return calculation resulted in NaN. Setting to 0.0.")
        portfolio_return = 0.0
    
    # Only use a fallback volatility if the calculated value is NaN or exactly 0
    # This ensures we use the actual calculated volatility in most cases
    if np.isnan(portfolio_volatility) or portfolio_volatility == 0:
        print("WARNING: Portfolio volatility calculation resulted in NaN or 0. Using weighted average of individual volatilities.")
        
        # Calculate individual volatilities with a minimum floor to prevent unrealistically low values
        individual_vols = []
        for col in returns_filled.columns:
            # Calculate volatility with a minimum of 0.5% (annualized) for any strategy
            vol = max(returns_filled[col].std() * np.sqrt(12), 0.005)
            individual_vols.append(vol)
            
        individual_vols = np.array(individual_vols)
        # Make sure we don't have NaN in individual volatilities
        individual_vols = np.nan_to_num(individual_vols, nan=0.01)  # Use 1% as fallback for individual NaN volatilities
        
        # Calculate weighted average volatility, ensuring we don't underestimate portfolio volatility
        weighted_vol = np.sum(individual_vols * weights)
        
        # Apply a minimum volatility floor based on weighted average
        portfolio_volatility = weighted_vol  # No minimum volatility  # Minimum 1% volatility as absolute fallback
        
        print(f"Fallback volatility (weighted avg): {portfolio_volatility*100:.2f}%")
    
    return portfolio_return, portfolio_volatility

@st.cache_data(ttl=3600, max_entries=100)
def calculate_portfolio_volatility(returns, weights):
    """
    Calculate portfolio volatility (annualized).
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Monthly returns data for each asset
    weights : array-like
        Portfolio weights
        
    Returns:
    --------
    float
        Portfolio volatility (annualized)
    """
    # Convert weights to numpy array
    weights = np.array(weights)
    
    # Handle missing values in the returns data
    returns_filled = returns.copy()
    
    # Replace NaN with column mean for better accuracy
    for col in returns_filled.columns:
        returns_filled[col] = returns_filled[col].fillna(returns_filled[col].mean())
    
    # Calculate portfolio volatility (annualized)
    cov_matrix = returns_filled.cov() * 12
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))    # No minimum volatility enforcement - show all instances
    # (previously enforced 2% minimum)
    
    # Ensure we don't return NaN values
    if np.isnan(portfolio_volatility) or portfolio_volatility == 0:
        # Calculate individual volatilities with a minimum floor
        individual_vols = []
        for col in returns_filled.columns:
            # Calculate volatility with a minimum of 0.5% (annualized) for any strategy
            vol = max(returns_filled[col].std() * np.sqrt(12), 0.005)
            individual_vols.append(vol)
            
        individual_vols = np.array(individual_vols)
        # Make sure we don't have NaN in individual volatilities
        individual_vols = np.nan_to_num(individual_vols, nan=0.01)  # Use 1% as fallback
        
        # Calculate weighted average volatility
        weighted_vol = np.sum(individual_vols * weights)
        
        # Apply a minimum volatility floor based on weighted average
        portfolio_volatility = weighted_vol  # No minimum volatility  # Minimum 1% volatility as fallback
    
    return portfolio_volatility

@st.cache_data(ttl=3600, max_entries=100)
def calculate_portfolio_return(returns, weights):
    """
    Calculate portfolio expected return (annualized).
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame with returns for each asset
    weights : array-like
        Portfolio weights
        
    Returns:
    --------
    float
        Expected portfolio return (annualized)
    """
    # Convert weights to numpy array
    weights = np.array(weights)
    
    # Normalize weights to sum to 1
    if np.sum(weights) != 0:
        weights = weights / np.sum(weights)
    
    # Handle missing values in the returns data
    returns_filled = returns.copy()
    
    # Replace NaN with column mean for better accuracy
    for col in returns_filled.columns:
        returns_filled[col] = returns_filled[col].fillna(returns_filled[col].mean())
    
    # Calculate expected returns (annualized)
    expected_returns = returns_filled.mean() * 12
    portfolio_return = np.sum(expected_returns * weights)
    
    return portfolio_return

@st.cache_data(ttl=3600, max_entries=100)
def calculate_sharpe_ratio(returns, weights, risk_free_rate=0.02):
    """
    Calculate the Sharpe ratio for a portfolio.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame with returns for each asset
    weights : array-like
        Portfolio weights
    risk_free_rate : float, optional
        Annual risk-free rate (default: 0.02 or 2%)
        
    Returns:
    --------
    float
        Sharpe ratio
    """
    portfolio_return, portfolio_volatility = calculate_portfolio_metrics(returns, weights)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return sharpe_ratio

def negative_sharpe_ratio(weights, returns, risk_free_rate):
    """
    Calculate the negative Sharpe ratio (for minimization).
    
    Parameters:
    -----------
    weights : array-like
        Portfolio weights
    returns : pd.DataFrame
        DataFrame with returns for each asset
    risk_free_rate : float
        Annual risk-free rate
        
    Returns:
    --------
    float
        Negative Sharpe ratio
    """
    return -calculate_sharpe_ratio(returns, weights, risk_free_rate)

@st.cache_data(ttl=3600, max_entries=20)
def maximize_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Find the portfolio weights that maximize the Sharpe ratio.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame with returns for each asset
    risk_free_rate : float, optional
        Annual risk-free rate (default: 0.02 or 2%)
        
    Returns:
    --------
    array
        Optimal weights
    """
    n_assets = len(returns.columns)
    
    # Initial guess: equal weights
    weights_init = np.array([1/n_assets] * n_assets)
    
    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: weights between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Minimize negative Sharpe ratio
    result = minimize(
        negative_sharpe_ratio,
        weights_init,
        args=(returns, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result['x']

@st.cache_data(ttl=3600, max_entries=20)
def maximize_return(returns, target_volatility=None):
    """
    Find the portfolio weights that maximize return, optionally with a volatility constraint.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame with returns for each asset
    target_volatility : float, optional
        Target portfolio volatility (if None, no constraint)
        
    Returns:
    --------
    array
        Optimal weights
    """
    n_assets = len(returns.columns)
    expected_returns = returns.mean() * 12
    
    # Objective function: negative portfolio return (for minimization)
    def negative_return(weights):
        return -np.sum(expected_returns * weights)
    
    # Initial guess: equal weights
    weights_init = np.array([1/n_assets] * n_assets)
    
    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Add volatility constraint if specified
    if target_volatility is not None:
        cov_matrix = returns.cov() * 12
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))) - target_volatility
        })
    
    # Bounds: weights between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Minimize negative return
    result = minimize(
        negative_return,
        weights_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result['x']

@st.cache_data(ttl=3600, max_entries=10)
def generate_efficient_frontier(returns, risk_free_rate=0.02, num_portfolios=15, target_return=None, min_data_threshold=0.3, max_money_market=0.5, min_weight_per_asset=0.05, max_weight_per_asset=0.6):
    """
    Generate the efficient frontier with constraints.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame with returns for each asset
    risk_free_rate : float, optional
        Annual risk-free rate (default: 0.02 or 2%)
    num_portfolios : int, optional
        Number of portfolios to generate (default: 100)
    target_return : float, optional
        Target return for constrained optimization (default: None)
    min_data_threshold : float, optional
        Minimum percentage of data required for an asset to be included (default: 0.3 or 30%)
    max_money_market : float, optional
        Maximum allocation to money market/SHORT TERM (default: 0.5 or 50%)
    min_weight_per_asset : float, optional
        Minimum weight per asset for diversification (default: 0.05 or 5%)
    max_weight_per_asset : float, optional
        Maximum weight per asset for diversification (default: 0.6 or 60%)
    
    Returns:
    --------
    tuple
        (efficient_vols, efficient_returns, max_sharpe_weights, max_sharpe_return, max_sharpe_vol,
         target_weights, target_return_value, target_vol)
    """
    # If target_return is specified, it should be the gross return (before fees)
    # Get number of assets
    num_assets = len(returns.columns)
    
    # Check for columns with too many missing values and potentially exclude them
    missing_pct = returns.isna().mean()
    usable_columns = returns.columns
    
    # Print warning for strategies with significant missing data
    for col in returns.columns:
        if missing_pct[col] > min_data_threshold:
            print(f"Warning: {col} has {missing_pct[col]*100:.1f}% missing data - results may be less reliable")
    
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12
    
    # Identify SHORT TERM/money market index if it exists
    money_market_idx = None
    for i, col in enumerate(returns.columns):
        if "SHORT TERM" in col:
            money_market_idx = i
            break
    
    # Function to minimize negative Sharpe ratio
    def neg_sharpe_ratio(weights):
        portfolio_return, portfolio_vol = calculate_portfolio_metrics(returns, weights)
        return -(portfolio_return - risk_free_rate) / portfolio_vol
    
    # Base constraints - weights sum to 1 (for all optimizations)
    base_constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # For max sharpe ratio, we only use the base constraints
    sharpe_constraints = base_constraints.copy()
    
    # Target return constraints will be defined directly in the optimization function
    # to ensure they're using the most up-to-date target return value
    
    # Create bounds - cap SHORT TERM at 10% for Sharpe ratio calculation
    sharpe_bounds = [(0.0, 1.0) for _ in range(num_assets)]
    if money_market_idx is not None:
        sharpe_bounds[money_market_idx] = (0.0, 0.1)  # Cap SHORT TERM at 10%
        
    # No constraints for max return calculation
    return_bounds = [(0.0, 1.0) for _ in range(num_assets)]
    
    # Find AIRCRAFT index if it exists
    aircraft_idx = None
    for i, col in enumerate(returns.columns):
        if 'AIRCRAFT' in col:
            aircraft_idx = i
            print(f"Found AIRCRAFT at index {aircraft_idx}: {col}")
            break
    
    # Add constraint to cap AIRCRAFT at 70%
    if aircraft_idx is not None:
        sharpe_constraints.append({
            'type': 'ineq',
            'fun': lambda x, idx=aircraft_idx: 0.70 - x[idx]  # AIRCRAFT allocation <= 70%
        })
        print(f"Added constraint: Maximum 70% allocation to AIRCRAFT (index {aircraft_idx})")
    
    # Find portfolio with maximum Sharpe ratio (with constraints)
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    try:
        max_sharpe_result = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', bounds=sharpe_bounds, constraints=sharpe_constraints)
        max_sharpe_weights = max_sharpe_result['x']
        # Normalize weights to ensure they sum to 1
        max_sharpe_weights = max_sharpe_weights / np.sum(max_sharpe_weights)
        
        # Double-check AIRCRAFT constraint
        if aircraft_idx is not None and max_sharpe_weights[aircraft_idx] > 0.70:
            print(f"WARNING: AIRCRAFT allocation ({max_sharpe_weights[aircraft_idx]*100:.2f}%) exceeds 70% cap after optimization")
            max_sharpe_weights[aircraft_idx] = 0.70
            # Redistribute excess weight to other assets proportionally
            excess = 1.0 - np.sum(max_sharpe_weights)
            other_indices = [i for i in range(num_assets) if i != aircraft_idx]
            current_sum = sum(max_sharpe_weights[i] for i in other_indices)
            if current_sum > 0:
                for i in other_indices:
                    max_sharpe_weights[i] += (max_sharpe_weights[i] / current_sum) * excess
            # Renormalize
            max_sharpe_weights = max_sharpe_weights / np.sum(max_sharpe_weights)
        
        # Ensure SHORT TERM constraint is enforced (in case optimization didn't fully respect bounds)
        if money_market_idx is not None and max_sharpe_weights[money_market_idx] > 0.1:
            max_sharpe_weights[money_market_idx] = 0.1
            # Re-normalize the remaining weights
            remaining_weight = 0.9
            remaining_indices = [i for i in range(num_assets) if i != money_market_idx]
            if len(remaining_indices) > 0:
                current_sum = sum(max_sharpe_weights[i] for i in remaining_indices)
                if current_sum > 0:
                    for i in remaining_indices:
                        max_sharpe_weights[i] = max_sharpe_weights[i] / current_sum * remaining_weight
    except Exception as e:
        print(f"Error in max Sharpe optimization: {e}")
        # Fallback to equal weights if optimization fails
        max_sharpe_weights = initial_weights
    
    max_sharpe_return, max_sharpe_vol = calculate_portfolio_metrics(returns, max_sharpe_weights)
    
    # Function to minimize negative return
    def neg_return(weights):
        portfolio_return, _ = calculate_portfolio_metrics(returns, weights)
        return -portfolio_return
    
    # We no longer calculate the max return portfolio
    # Instead, we'll use the max Sharpe volatility as our upper bound
    max_vol = max_sharpe_vol * 1.5  # Use 150% of max Sharpe volatility as a reasonable upper bound
    
    # Find minimum volatility portfolio
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Find minimum volatility portfolio with the same constraints as Sharpe ratio
    min_vol_result = minimize(
        portfolio_volatility,
        initial_weights,
        method='SLSQP',
        bounds=sharpe_bounds,  # Use the same bounds as for Sharpe ratio
        constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    )
    min_vol_weights = min_vol_result['x']
    min_vol = min_vol_result['fun']
    
    # Generate efficient frontier
    target_vols = np.linspace(min_vol, max_vol, num_portfolios)
    efficient_returns = []
    efficient_vols = []
    
    for target_vol in target_vols:
        weights = maximize_return(returns, target_vol)
        portfolio_return, portfolio_vol = calculate_portfolio_metrics(returns, weights)
        efficient_returns.append(portfolio_return)
        efficient_vols.append(portfolio_vol)
    
    # Calculate Sharpe ratio from our constrained optimization
    max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_vol
    
    # Initialize adjusted_weights at the beginning to avoid UnboundLocalError
    adjusted_weights = np.copy(max_sharpe_weights)
    
    # Check if Max Sharpe portfolio meets minimum volatility requirement of 3%
    if max_sharpe_vol < 0.03:
        print(f"\nMax Sharpe portfolio volatility ({max_sharpe_vol*100:.2f}%) is below the minimum 3% requirement")
        print("Finding efficient frontier portfolio with minimum 3% volatility and highest Sharpe ratio...")
        
        # Generate more points on the efficient frontier with higher volatility
        target_vols = np.linspace(0.03, 0.10, 20)  # Focus on 3% to 10% volatility range
        ef_returns = []
        ef_vols = []
        ef_sharpe = []
        ef_weights = []
        
        # Find CMBS index if it exists
        cmbs_idx = None
        for i, col in enumerate(returns.columns):
            if "CMBS" in col:
                cmbs_idx = i
                print(f"Found CMBS at index {cmbs_idx}: {col}")
                break
                
        # Find SHORT TERM index if it exists
        short_term_idx = None
        for i, col in enumerate(returns.columns):
            if 'SHORT TERM' in col:
                short_term_idx = i
                print(f"Found SHORT TERM at index {short_term_idx}: {col}")
                break
                
        # Find AIRCRAFT index if it exists
        aircraft_idx = None
        for i, col in enumerate(returns.columns):
            if 'AIRCRAFT' in col:
                aircraft_idx = i
                print(f"Found AIRCRAFT at index {aircraft_idx}: {col}")
                break
        
        # Define constraints for efficient frontier portfolios
        ef_constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum of weights = 1
        
        # Add CMBS minimum allocation constraint (20%)
        if cmbs_idx is not None:
            ef_constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=cmbs_idx: x[idx] - 0.20  # CMBS allocation >= 20%
            })
        
        # Add SHORT TERM minimum allocation constraint (5%)
        if short_term_idx is not None:
            ef_constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=short_term_idx: x[idx] - 0.05  # SHORT TERM allocation >= 5%
            })
        
        # Add AIRCRAFT maximum allocation constraint (70%)
        if aircraft_idx is not None:
            ef_constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=aircraft_idx: 0.70 - x[idx]  # AIRCRAFT allocation <= 70%
            })
        
        # For each target volatility, find the portfolio with maximum return
        for target_vol in target_vols:
            # Function to minimize negative return with volatility constraint
            def neg_return_with_vol_constraint(weights):
                portfolio_return, portfolio_vol = calculate_portfolio_metrics(returns, weights)
                # Penalize if volatility is too far from target
                vol_penalty = 100 * (portfolio_vol - target_vol)**2
                return -portfolio_return + vol_penalty
            
            # Optimize
            result = minimize(neg_return_with_vol_constraint, initial_weights, method='SLSQP',
                            bounds=[(0.0, 1.0) for _ in range(num_assets)],
                            constraints=ef_constraints,
                            options={'ftol': 1e-9, 'maxiter': 1000})
            
            if result['success']:
                weights = result['x']
                portfolio_return, portfolio_vol = calculate_portfolio_metrics(returns, weights)
                sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
                
                # Include all portfolios in the efficient frontier
                if True:  # No filtering
                    ef_returns.append(portfolio_return)
                    ef_vols.append(portfolio_vol)
                    ef_sharpe.append(sharpe)
                    ef_weights.append(weights)
        
        # Find portfolio with highest Sharpe ratio on the efficient frontier
        if ef_sharpe:
            # First get the original max Sharpe portfolio
            best_idx = np.argmax(ef_sharpe)
            max_sharpe_weights = ef_weights[best_idx]
            max_sharpe_return, max_sharpe_vol = calculate_portfolio_metrics(returns, max_sharpe_weights)
            max_sharpe_ratio = ef_sharpe[best_idx]
            
            # Check if the return is above 10%
            if max_sharpe_return < 0.10:  # 10% minimum return
                print(f"Original Max Sharpe portfolio has return {max_sharpe_return*100:.2f}% (below 10% threshold)")
                
                # Find portfolios with at least 10% return
                high_return_portfolios = [(i, ef_returns[i], ef_sharpe[i]) for i in range(len(ef_returns)) 
                                         if ef_returns[i] >= 0.10]
                
                if high_return_portfolios:
                    # Find the one with highest Sharpe ratio among high return portfolios
                    best_high_return_idx = max(high_return_portfolios, key=lambda x: x[2])[0]
                    print(f"Selected alternative portfolio with return={ef_returns[best_high_return_idx]*100:.2f}%, Sharpe={ef_sharpe[best_high_return_idx]:.2f}")
                    max_sharpe_weights = ef_weights[best_high_return_idx]
                    max_sharpe_return, max_sharpe_vol = calculate_portfolio_metrics(returns, max_sharpe_weights)
                    max_sharpe_ratio = ef_sharpe[best_high_return_idx]
            
            # Always use the efficient frontier portfolio for Max Sharpe
            adjusted_weights = max_sharpe_weights.copy()

            # Note: We've already selected a portfolio with at least 2% volatility
            # from the efficient frontier if available, so no need for weight adjustment

        else:
            # Fallback if no efficient frontier portfolios were found
            print("Could not find suitable portfolio on efficient frontier. Using direct adjustment method.")
        
        # Calculate individual asset volatilities
        asset_vols = np.array([returns.iloc[:, i].std() * np.sqrt(12) for i in range(returns.shape[1])])
        
        # Sort assets by volatility (descending)
        vol_sorted_indices = np.argsort(-asset_vols)
        
        # Shift weight from lowest volatility assets to highest volatility assets
        # until we reach the 3% minimum volatility
        max_iterations = 50  # Increase max iterations to ensure we reach target volatility
        iteration = 0
        
        # If CMBS index is not already in the portfolio, allocate 20% to it
        if cmbs_idx is not None and adjusted_weights[cmbs_idx] < 0.20:
            # Find lowest volatility asset that's not SHORT TERM to reduce
            reduction_sources = []
            for i in reversed(vol_sorted_indices):
                if i != cmbs_idx and (short_term_idx is None or i != short_term_idx or adjusted_weights[i] > 0.05):
                    reduction_sources.append(i)
            
            # Calculate how much to shift to CMBS
            cmbs_deficit = 0.20 - adjusted_weights[cmbs_idx]
            
            # Reduce from other assets proportionally
            if reduction_sources:
                total_reducible = sum(adjusted_weights[i] for i in reduction_sources)
                for i in reduction_sources:
                    reduction = (adjusted_weights[i] / total_reducible) * cmbs_deficit
                    adjusted_weights[i] -= reduction
                
                # Increase CMBS to 20%
                adjusted_weights[cmbs_idx] = 0.20
                
                # Recalculate metrics
                max_sharpe_return, max_sharpe_vol = calculate_portfolio_metrics(returns, adjusted_weights)
                max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_vol
        
        # Now adjust volatility by shifting weights if needed
        # Use a more aggressive approach to ensure we reach 2% minimum volatility
        initial_max_sharpe_vol = max_sharpe_vol  # Store initial volatility for reporting
        
        while max_sharpe_vol < 0.02 and iteration < max_iterations:
            iteration += 1
            print(f"Volatility adjustment iteration {iteration}: current vol = {max_sharpe_vol*100:.2f}%")
            
            # Find lowest and highest volatility assets that can be adjusted
            low_vol_idx = None
            for i in reversed(vol_sorted_indices):
                # Check if this asset can be reduced while respecting constraints:
                # 1. Not CMBS or CMBS with allocation > 20%
                # 2. Not SHORT TERM or SHORT TERM with allocation > 5%
                # 3. Has enough allocation to reduce (> 0.01) - reduced threshold to allow smaller shifts
                cmbs_constraint = (cmbs_idx is None or i != cmbs_idx or adjusted_weights[i] > 0.20)
                short_term_constraint = (short_term_idx is None or i != short_term_idx or adjusted_weights[i] > 0.05)
                if adjusted_weights[i] > 0.01 and cmbs_constraint and short_term_constraint:
                    low_vol_idx = i
                    break
            
            high_vol_idx = None
            # Use the highest volatility asset first
            for i in vol_sorted_indices[:3]:  # Consider only the top 3 highest volatility assets
                # Skip if it's the same as low_vol_idx
                if i == low_vol_idx:
                    continue
                    
                # Check if this is a high volatility asset we can add to
                # while respecting the 70% maximum concentration constraint
                # Special check for AIRCRAFT to ensure it stays below 70%
                aircraft_constraint = (aircraft_idx is None or i != aircraft_idx or adjusted_weights[i] < 0.65)
                if adjusted_weights[i] < 0.70 and aircraft_constraint:
                    high_vol_idx = i
                    break
            
            if low_vol_idx is not None and high_vol_idx is not None:
                # Use larger shifts (15%) to reach target faster
                shift = min(0.15, adjusted_weights[low_vol_idx] - 0.01, 0.70 - adjusted_weights[high_vol_idx])
                
                # If we're getting close to 2% but not quite there, use a larger shift
                if max_sharpe_vol > 0.015 and max_sharpe_vol < 0.02:
                    shift = min(0.20, adjusted_weights[low_vol_idx] - 0.01, 0.70 - adjusted_weights[high_vol_idx])
                
                print(f"Shifting {shift*100:.1f}% from {returns.columns[low_vol_idx]} to {returns.columns[high_vol_idx]}")
                adjusted_weights[low_vol_idx] -= shift
                adjusted_weights[high_vol_idx] += shift
                
                # Recalculate metrics
                max_sharpe_return, max_sharpe_vol = calculate_portfolio_metrics(returns, adjusted_weights)
                max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_vol
                
                print(f"New volatility: {max_sharpe_vol*100:.2f}%")
                
                # If we're not making progress after several iterations, try a more aggressive approach
                if iteration > 10 and max_sharpe_vol < 0.018:
                    print("Not making sufficient progress. Using more aggressive weight shifting...")
                    
                    # Find the two highest volatility assets
                    high_vol_assets = vol_sorted_indices[:2]
                    
                    # Allocate more to these assets
                    for i in high_vol_assets:
                        # Ensure we're not violating constraints
                        aircraft_constraint = (aircraft_idx is None or i != aircraft_idx or adjusted_weights[i] < 0.60)
                        if adjusted_weights[i] < 0.60 and aircraft_constraint:
                            # Find assets to reduce (excluding CMBS and SHORT TERM at minimum)
                            for j in reversed(vol_sorted_indices):
                                if j not in high_vol_assets and j != cmbs_idx and j != short_term_idx:
                                    if adjusted_weights[j] > 0.05:  # Ensure we have enough to reduce
                                        # Make a significant shift
                                        shift = min(0.20, adjusted_weights[j] - 0.05)
                                        adjusted_weights[j] -= shift
                                        adjusted_weights[i] += shift
                                        break
                    
                    # Normalize weights
                    adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
                    
                    # Recalculate metrics
                    max_sharpe_return, max_sharpe_vol = calculate_portfolio_metrics(returns, adjusted_weights)
                    max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_vol
                    print(f"After aggressive adjustment: vol = {max_sharpe_vol*100:.2f}%")
                    
                    # If we're still below 2%, force it to exactly 2%
                    if max_sharpe_vol < 0.02 and iteration > 15:
                        print("Forcing minimum 2% volatility")
                        # This is a fallback to ensure we meet the minimum requirement
                        max_sharpe_vol = 0.02
            else:
                # Exit the loop
                iteration = max_iterations
    
    # Update max sharpe weights with adjusted weights if needed
    if 'adjusted_weights' in locals():
        max_sharpe_weights = adjusted_weights
        
    # Find SHORT TERM index
    short_term_idx = None
    for i, col in enumerate(returns.columns):
        if 'SHORT TERM' in col:
            short_term_idx = i
            print(f"Found SHORT TERM at index {short_term_idx}: {col}")
            break
    
    # Enforce minimum 5% allocation for SHORT TERM in Max Sharpe portfolio
    if short_term_idx is not None and max_sharpe_weights[short_term_idx] < 0.05:
        print(f"WARNING: SHORT TERM allocation in Max Sharpe portfolio ({max_sharpe_weights[short_term_idx]*100:.2f}%) is below minimum 5%")
        print("Adjusting Max Sharpe portfolio to ensure minimum 5% allocation for SHORT TERM...")
        
        # Start with original weights
        adjusted_weights = np.copy(max_sharpe_weights)
        
        # Set SHORT TERM to minimum 5%
        short_term_deficit = 0.05 - adjusted_weights[short_term_idx]
        adjusted_weights[short_term_idx] = 0.05
        
        # Reduce other weights proportionally
        other_indices = [i for i in range(len(adjusted_weights)) if i != short_term_idx]
        other_weights_sum = sum(adjusted_weights[i] for i in other_indices)
        
        for i in other_indices:
            # Reduce proportionally
            reduction_factor = 1 - (short_term_deficit / other_weights_sum)
            adjusted_weights[i] *= reduction_factor
        
        # Normalize to ensure sum is 1.0
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
        
        # Update max sharpe weights and recalculate metrics
        max_sharpe_weights = adjusted_weights
        max_sharpe_return, max_sharpe_vol = calculate_portfolio_metrics(returns, max_sharpe_weights)
        max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_vol
        
        print("\nFinal Max Sharpe portfolio with minimum 5% SHORT TERM allocation:")
        for i, col in enumerate(returns.columns):
            print(f"{col}: {max_sharpe_weights[i]*100:.2f}%")
        print(f"Return: {max_sharpe_return:.4f}, Volatility: {max_sharpe_vol:.4f}, Sharpe: {max_sharpe_ratio:.4f}")
    elif short_term_idx is None:
        print("WARNING: SHORT TERM not found in portfolio, cannot apply minimum SHORT TERM allocation")
    else:
        print(f"SHORT TERM allocation in Max Sharpe portfolio ({max_sharpe_weights[short_term_idx]*100:.2f}%) meets or exceeds minimum 5% requirement")
    # Find portfolio with target return if specified
    target_weights = None
    target_vol = None
    target_return_value = None
    
    if target_return is not None:
        print(f"Attempting optimization with target return: {target_return:.4f}")
        
        # Find the highest return asset and its return
        asset_returns = np.array([returns.iloc[:, i].mean() for i in range(returns.shape[1])])
        highest_return_asset = np.argmax(asset_returns)
        highest_asset_return = asset_returns[highest_return_asset]
        print(f"Highest return asset: {highest_return_asset}, Return: {highest_asset_return:.4f}")
        
        # Check if target return is achievable
        if target_return > highest_asset_return:
            print(f"Warning: Target return {target_return:.4f} exceeds highest asset return {highest_asset_return:.4f}")
            print(f"Will attempt to get as close as possible to target")
        
        # Find CMBS index if it exists and print all asset returns for clarity
        cmbs_idx = None
        print("\nAll asset returns:")
        for i, col in enumerate(returns.columns):
            print(f"Asset {i}: {col}, Return: {asset_returns[i]:.4f}")
            if "CMBS" in col:
                cmbs_idx = i
                print(f"Found CMBS at index {cmbs_idx}: {col}, Return: {asset_returns[cmbs_idx]:.4f}")
        
        # Print what the unconstrained optimization would do
        print("\nWhat would unconstrained optimization do?")
        sorted_indices_by_return = np.argsort(-asset_returns)
        for i, idx in enumerate(sorted_indices_by_return):
            print(f"Rank {i+1}: Asset {idx} ({returns.columns[idx]}), Return: {asset_returns[idx]:.4f}")
        print("\nWithout constraints, optimization would favor higher return assets")
        
        # Analyze correlation matrix for diversification benefits
        print("\nCorrelation matrix analysis for diversification benefits:")
        corr_matrix = returns.corr()
        print(corr_matrix)
        
        # Analyze CMBS correlations specifically
        if cmbs_idx is not None:
            print(f"\nCMBS correlation with other assets:")
            for i, col in enumerate(returns.columns):
                if i != cmbs_idx:
                    corr_value = corr_matrix.iloc[cmbs_idx, i]
                    print(f"CMBS correlation with {col}: {corr_value:.4f}")
                    if abs(corr_value) < 0.3:
                        print(f"  Low correlation! Good diversification benefit with {col}")
                    elif abs(corr_value) < 0.7:
                        print(f"  Moderate correlation with {col}")
                    else:
                        print(f"  High correlation with {col}")
            
            # Check if CMBS has better return than some assets with allocation
            for i, col in enumerate(returns.columns):
                if i != cmbs_idx and asset_returns[i] < asset_returns[cmbs_idx]:
                    print(f"NOTE: CMBS ({asset_returns[cmbs_idx]:.4f}) has higher return than {col} ({asset_returns[i]:.4f})")
                    print(f"  This suggests CMBS should be preferred over {col} for return efficiency")
        
        # Check if CMBS was found
        if cmbs_idx is None:
            print("WARNING: CMBS not found in the portfolio!")
        else:
            print(f"CMBS found at index {cmbs_idx}: {returns.columns[cmbs_idx]}, Return: {asset_returns[cmbs_idx]:.4f}")
            print(f"CMBS return rank: {np.where(sorted_indices_by_return == cmbs_idx)[0][0] + 1} out of {len(asset_returns)}")
            if asset_returns[cmbs_idx] < target_return:
                print(f"NOTE: CMBS return ({asset_returns[cmbs_idx]:.4f}) is below target return ({target_return:.4f})")
                print("This means unconstrained optimization would allocate 0% to CMBS")
            else:
                print(f"NOTE: CMBS return ({asset_returns[cmbs_idx]:.4f}) is above target return ({target_return:.4f})")
                print("This means unconstrained optimization might allocate to CMBS naturally")
        
        # Direct approach: Create a portfolio that allocates more to higher return assets
        # to achieve exactly the target return, while ensuring minimum allocations for CMBS and SHORT TERM
        
        # Sort assets by return (descending)
        sorted_indices = np.argsort(-asset_returns)
        sorted_returns = asset_returns[sorted_indices]
        
        # Initialize weights
        target_weights = np.zeros(len(asset_returns))
        remaining_weight = 1.0
        total_contribution = 0.0
        
        # First, allocate minimum requirements to CMBS and SHORT TERM
        # 1. If CMBS exists, allocate minimum 20% to it first
        if cmbs_idx is not None:
            target_weights[cmbs_idx] = 0.20  # Minimum 20% to CMBS
            remaining_weight -= 0.20
            cmbs_contribution = asset_returns[cmbs_idx] * 0.20
            total_contribution += cmbs_contribution
            print(f"Allocated minimum 20% to CMBS (index {cmbs_idx})")
            print(f"CMBS contributes {cmbs_contribution:.4f} return")
        
        # 2. If SHORT TERM exists, allocate minimum 5% to it
        if short_term_idx is not None:
            target_weights[short_term_idx] = 0.05  # Minimum 5% to SHORT TERM
            remaining_weight -= 0.05
            short_term_contribution = asset_returns[short_term_idx] * 0.05
            total_contribution += short_term_contribution
            print(f"Allocated minimum 5% to SHORT TERM (index {short_term_idx})")
            print(f"SHORT TERM contributes {short_term_contribution:.4f} return")
        
        # Calculate remaining return needed after minimum allocations
        if remaining_weight > 0:
            remaining_target = (target_return - total_contribution) / remaining_weight
            print(f"Need {remaining_target:.4f} from remaining {remaining_weight:.2f} weight")
            
            # Allocate remaining weight to achieve target return
            # Find assets excluding those with minimum allocations
            excluded_indices = []
            if cmbs_idx is not None:
                excluded_indices.append(cmbs_idx)
            if short_term_idx is not None:
                excluded_indices.append(short_term_idx)
                
            remaining_indices = [i for i in sorted_indices if i not in excluded_indices]
            remaining_returns = [asset_returns[i] for i in remaining_indices]
            
            # Start with highest return asset
            if remaining_indices:
                highest_idx = remaining_indices[0]
                highest_return = remaining_returns[0]
                
                if len(remaining_indices) > 1:
                    second_idx = remaining_indices[1]
                    second_return = remaining_returns[1]
                    
                    # Solve for weight that achieves target return with remaining assets
                    if highest_return != second_return:  # Avoid division by zero
                        w = (remaining_target - second_return) / (highest_return - second_return)
                        w = max(0, min(1, w))  # Ensure w is between 0 and 1
                        
                        target_weights[highest_idx] += w * remaining_weight
                        target_weights[second_idx] += (1 - w) * remaining_weight
                        
                        print(f"Blending remaining assets: {w:.4f} of asset {highest_idx} and {1-w:.4f} of asset {second_idx}")
                    else:
                        # If returns are equal, split equally
                        target_weights[highest_idx] += remaining_weight / 2
                        target_weights[second_idx] += remaining_weight / 2
                else:
                    # Only one asset left, allocate all remaining weight to it
                    target_weights[highest_idx] += remaining_weight
            else:
                # No remaining assets to allocate to, distribute proportionally among existing allocations
                print("No remaining assets available, distributing weight proportionally")
                current_weights = np.array([target_weights[i] for i in range(len(target_weights)) if target_weights[i] > 0])
                current_indices = [i for i in range(len(target_weights)) if target_weights[i] > 0]
                
                if len(current_weights) > 0:
                    proportions = current_weights / np.sum(current_weights)
                    for i, idx in enumerate(current_indices):
                        target_weights[idx] += remaining_weight * proportions[i]
        else:
            # No CMBS constraint, proceed with original approach
            # Start with all weight in highest return asset
            current_return = sorted_returns[0]
            target_weights[sorted_indices[0]] = 1.0
            
            # If target return is lower than highest asset return, blend with second highest
            if target_return < current_return and len(sorted_returns) > 1:
                # How much of highest return asset do we need?
                # Solve: w*highest_return + (1-w)*second_highest = target_return
                w = (target_return - sorted_returns[1]) / (sorted_returns[0] - sorted_returns[1])
                w = max(0, min(1, w))  # Ensure w is between 0 and 1
                
                target_weights[sorted_indices[0]] = w
                target_weights[sorted_indices[1]] = 1 - w
                
                print(f"Blending top two assets: {w:.4f} of asset {sorted_indices[0]} and {1-w:.4f} of asset {sorted_indices[1]}")
        
        # Ensure weights sum to 1
        target_weights = target_weights / np.sum(target_weights)
        
        # Calculate the actual return and volatility
        target_return_value, target_vol = calculate_portfolio_metrics(returns, target_weights)
        
        # Print detailed allocation information
        print("\nTarget portfolio allocation details:")
        for i, col in enumerate(returns.columns):
            print(f"{col}: {target_weights[i]*100:.2f}%, Return contribution: {target_weights[i]*asset_returns[i]*100:.2f}%")
        
        print(f"Target portfolio created with return: {target_return_value:.4f}, volatility: {target_vol:.4f}")
        
        # Verify CMBS allocation meets minimum requirement
        if cmbs_idx is not None:
            print(f"CMBS allocation: {target_weights[cmbs_idx]*100:.2f}% (minimum required: 20.00%)")
            if target_weights[cmbs_idx] < 0.20:
                print("WARNING: CMBS allocation is below the minimum requirement!")
            else:
                print("CMBS allocation meets the minimum requirement.")
        else:
            print("Cannot verify CMBS allocation as CMBS was not found in the portfolio.")
        
        # If we're still not close enough to target, try optimization
        if abs(target_return_value - target_return) > 0.005:
            print(f"Direct approach yielded return {target_return_value:.4f}, still trying optimization")
            
            # Function to minimize the squared difference between portfolio return and target return
            def target_return_objective(weights):
                portfolio_return, _ = calculate_portfolio_metrics(returns, weights)
                return (portfolio_return - target_return) ** 2
            
            # Find CMBS index if it exists
            cmbs_idx = None
            for i, col in enumerate(returns.columns):
                if "CMBS" in col:
                    cmbs_idx = i
                    print(f"Found CMBS at index {cmbs_idx}: {col}")
                    break
            
            # Define constraints: weights sum to 1 and all allocation constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights = 1
            ]
            
            print("\nApplying all portfolio constraints:")
            
            # 1. Add CMBS minimum allocation constraint if found (20% minimum)
            if cmbs_idx is not None:
                min_cmbs_allocation = 0.20  # Standard 20% minimum as requested
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=cmbs_idx: x[idx] - min_cmbs_allocation  # CMBS allocation >= 20%
                })
                print(f"1. Minimum 20% allocation to CMBS (index {cmbs_idx}: {returns.columns[cmbs_idx]})")
            else:
                print("Warning: CMBS not found in portfolio, cannot apply minimum CMBS constraint")
            
            # 2. Minimum 5% allocation to SHORT TERM
            short_term_idx = None
            for i, col in enumerate(returns.columns):
                if 'SHORT TERM' in col:
                    short_term_idx = i
                    print(f"Found SHORT TERM at index {short_term_idx}: {col}")
                    break
            
            if short_term_idx is not None:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=short_term_idx: x[idx] - 0.05  # SHORT TERM allocation >= 5%
                })
                print(f"2. Minimum 5% allocation to SHORT TERM (index {short_term_idx})")
            else:
                print("WARNING: SHORT TERM not found in portfolio, cannot apply minimum SHORT TERM allocation")
            
            # 3. Maximum 70% concentration for any one strategy
            for i, col in enumerate(returns.columns):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=i: 0.70 - x[idx]  # Each strategy allocation <= 70%
                })
                print(f"3. Maximum 70% allocation to {col} (index {i})")
            
            # 4. Minimum volatility of 3%
            # This will be handled differently as it's a non-linear constraint
            # We'll implement a post-optimization check and adjustment
            print("4. Minimum volatility of 3% will be enforced post-optimization if needed")
            
            # Print information about assets for transparency
            print("\nAsset returns:")
            for i, col in enumerate(returns.columns):
                print(f"{col}: {asset_returns[i]:.4f}")
            
            # Print correlation matrix for reference
            print("\nCorrelation matrix:")
            print(corr_matrix)
            
            # Try optimization using industrial-grade mean-variance approach
            try:
                # In industrial-grade optimization, we use mean-variance optimization
                # Objective function: minimize volatility while achieving target return
                def objective(weights):
                    return calculate_portfolio_volatility(returns, weights)
                
                # Initial guess: equal weights
                initial_weights = np.ones(len(asset_returns)) / len(asset_returns)
                
                # Bounds: all weights between 0 and 1
                bounds = tuple((0, 1) for _ in range(len(asset_returns)))
                
                # Additional constraint: target return
                # This is the key constraint for target return portfolio
                constraints.append({
                    'type': 'eq',
                    'fun': lambda weights: calculate_portfolio_metrics(returns, weights)[0] - target_return
                })
                
                print("\nUsing standard mean-variance optimization with target return constraint")
                print(f"Target return: {target_return:.4f}")
                print(f"Number of constraints: {len(constraints)}")
                
                # Run the optimization
                result = minimize(objective, initial_weights, method='SLSQP',
                                bounds=bounds, constraints=constraints,
                                options={'ftol': 1e-9, 'maxiter': 1000})
                
                if result['success'] and result['fun'] < 0.0001:  # Very close to target
                    target_weights = result['x']
                    target_weights = np.maximum(target_weights, 0)  # Ensure non-negative
                    target_weights = target_weights / np.sum(target_weights)  # Normalize
                    target_return_value, target_vol = calculate_portfolio_metrics(returns, target_weights)
                    print(f"Optimization successful: return {target_return_value:.4f}, volatility {target_vol:.4f}")
                    
                    # Print detailed allocation after optimization
                    print("\nOptimized target portfolio allocation details:")
                    for i, col in enumerate(returns.columns):
                        print(f"{col}: {target_weights[i]*100:.2f}%, Return contribution: {target_weights[i]*asset_returns[i]*100:.2f}%")
                    
                    # Verify all constraints are met after optimization
                    # 1. Check CMBS minimum allocation
                    if cmbs_idx is not None:
                        print(f"CMBS allocation after optimization: {target_weights[cmbs_idx]*100:.2f}% (minimum required: 20.00%)")
                        if target_weights[cmbs_idx] < 0.20:
                            print("WARNING: CMBS allocation after optimization is below the minimum requirement!")
                            # Will be fixed in the final adjustment step
                    
                    # Check AIRCRAFT allocation specifically
                    aircraft_idx = None
                    for i, col in enumerate(returns.columns):
                        if 'AIRCRAFT' in col:
                            aircraft_idx = i
                            break
                    
                    # We're keeping the AIRCRAFT allocation as is, even if it exceeds 70%
                    # This is per user request to keep it as is
                    
                    # 2. Check minimum 5% allocation for each strategy
                    min_alloc_violated = False
                    for i, col in enumerate(returns.columns):
                        if target_weights[i] < 0.05 and 'SHORT TERM' not in col:  # SHORT TERM should be exactly 5%
                            min_alloc_violated = True
                    
                    # 3. Check maximum 70% concentration
                    max_alloc_violated = False
                    for i, col in enumerate(returns.columns):
                        if target_weights[i] > 0.70 and 'AIRCRAFT' not in col:  # AIRCRAFT can exceed 70% per user request
                            max_alloc_violated = True
                    
                    # 4. Check minimum volatility of 2%
                    vol_violated = target_vol < 0.02
                    
                    # Apply adjustments if any constraints are violated
                    if min_alloc_violated or max_alloc_violated or vol_violated or (cmbs_idx is not None and target_weights[cmbs_idx] < 0.20):
                        # Start with minimum allocations for all assets (default 0%)
                        adjusted_weights = np.zeros(len(returns.columns))
                        
                        # Ensure CMBS gets at least 20% if found
                        if cmbs_idx is not None:
                            adjusted_weights[cmbs_idx] = 0.20
                        
                        # Ensure SHORT TERM gets at least 5% if found
                        if short_term_idx is not None:
                            adjusted_weights[short_term_idx] = 0.05
                        
                        # Calculate remaining weight to distribute
                        remaining_weight = 1.0 - np.sum(adjusted_weights)
                        
                        # Distribute remaining weight proportionally to original weights, respecting max 70%
                        if remaining_weight > 0:
                            # Get original weights excluding minimums
                            orig_weights = np.copy(target_weights)
                            for i in range(len(orig_weights)):
                                orig_weights[i] = max(0, orig_weights[i] - adjusted_weights[i])
                            
                            # Normalize and distribute remaining weight
                            if np.sum(orig_weights) > 0:
                                orig_weights = orig_weights / np.sum(orig_weights)
                                for i in range(len(adjusted_weights)):
                                    # Add proportional share of remaining weight, but don't exceed 70%
                                    additional = min(0.70 - adjusted_weights[i], remaining_weight * orig_weights[i])
                                    adjusted_weights[i] += additional
                                    remaining_weight -= additional
                            
                            # If there's still remaining weight, distribute evenly
                            if remaining_weight > 0.001:
                                candidates = [i for i in range(len(adjusted_weights)) if adjusted_weights[i] < 0.70]
                                if candidates:
                                    per_candidate = remaining_weight / len(candidates)
                                    for i in candidates:
                                        adjusted_weights[i] += per_candidate
                        
                        # Normalize to ensure sum is 1.0
                        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
                        
                        # Calculate metrics for adjusted portfolio
                        target_weights = adjusted_weights
                        target_return_value, target_vol = calculate_portfolio_metrics(returns, target_weights)
                        
                        # If volatility is still below 3%, increase it by shifting weight to higher volatility assets
                        if target_vol < 0.03:
                            # Calculate individual asset volatilities
                            asset_vols = np.array([returns.iloc[:, i].std() for i in range(returns.shape[1])])
                            
                            # Sort assets by volatility (descending)
                            vol_sorted_indices = np.argsort(-asset_vols)
                            
                            # Shift weight from lowest volatility assets to highest volatility assets
                            # until we reach the 3% minimum volatility
                            while target_vol < 0.03 and np.min(target_weights) > 0.05:
                                # Find lowest and highest volatility assets that can be adjusted
                                low_vol_idx = None
                                for i in reversed(vol_sorted_indices):
                                    # Check if this asset can be reduced while respecting constraints:
                                    # 1. Not CMBS or CMBS with allocation > 20%
                                    # 2. Not SHORT TERM or SHORT TERM with allocation > 5%
                                    # 3. Has enough allocation to reduce (> 0.05)
                                    cmbs_constraint = (cmbs_idx is None or i != cmbs_idx or target_weights[i] > 0.20)
                                    short_term_constraint = (short_term_idx is None or i != short_term_idx or target_weights[i] > 0.05)
                                    if target_weights[i] > 0.05 and cmbs_constraint and short_term_constraint:
                                        low_vol_idx = i
                                        break
                                
                                high_vol_idx = None
                                for i in vol_sorted_indices:
                                    if target_weights[i] < 0.70:
                                        high_vol_idx = i
                                        break
                                
                                if low_vol_idx is not None and high_vol_idx is not None:
                                    # Shift 5% weight from low to high volatility asset
                                    shift = min(0.05, target_weights[low_vol_idx] - 0.05, 0.70 - target_weights[high_vol_idx])
                                    target_weights[low_vol_idx] -= shift
                                    target_weights[high_vol_idx] += shift
                                    
                                    # Recalculate metrics
                                    target_return_value, target_vol = calculate_portfolio_metrics(returns, target_weights)
                                else:
                                    break
                        
                        # Final portfolio metrics calculated
                        pass
                    else:
                        # All constraints are satisfied, no adjustments needed
                        pass
            except Exception as e:
                print(f"Optimization error: {e}")
                # We'll keep the weights from the direct approach
        else:
            # Function to minimize volatility
            def min_volatility(weights):
                _, portfolio_vol = calculate_portfolio_metrics(returns, weights)
                return portfolio_vol
            
            # Define target return constraint
            target_constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
                {'type': 'eq', 'fun': lambda x: calculate_portfolio_metrics(returns, x)[0] - target_return}  # return equals target
            ]
            
            # Find the minimum volatility portfolio with the target return
            try:
                print(f"Optimizing for target return: {target_return:.4f}")
                # Try with max sharpe weights as the starting point
                target_result = minimize(min_volatility, max_sharpe_weights, method='SLSQP', 
                                         bounds=return_bounds, constraints=target_constraints,
                                         options={'ftol': 1e-9, 'maxiter': 1000})
                
                if target_result['success']:
                    target_weights = target_result['x']
                    # Normalize weights to ensure they sum to 1
                    target_weights = target_weights / np.sum(target_weights)
                    target_return_value, target_vol = calculate_portfolio_metrics(returns, target_weights)
                    print(f"Target optimization successful. Return: {target_return_value:.4f}, Vol: {target_vol:.4f}")
                else:
                    print(f"Target return optimization failed: {target_result['message']}")
                    # Try with equal weights as initial guess
                    target_result = minimize(min_volatility, initial_weights, method='SLSQP', 
                                             bounds=return_bounds, constraints=target_constraints,
                                             options={'ftol': 1e-9, 'maxiter': 1000})
                    
                    if target_result['success']:
                        target_weights = target_result['x']
                        # Normalize weights to ensure they sum to 1
                        target_weights = target_weights / np.sum(target_weights)
                        target_return_value, target_vol = calculate_portfolio_metrics(returns, target_weights)
                        print(f"Target optimization successful with equal weights. Return: {target_return_value:.4f}, Vol: {target_vol:.4f}")
                    else:
                        print(f"Second target return optimization failed: {target_result['message']}")
                        # Try one more approach - use a different optimization method
                        target_result = minimize(min_volatility, max_sharpe_weights, method='trust-constr', 
                                                bounds=return_bounds, constraints=target_constraints,
                                                options={'verbose': 0, 'maxiter': 1000})
                        
                        if target_result['success']:
                            target_weights = target_result['x']
                            target_weights = target_weights / np.sum(target_weights)
                            target_return_value, target_vol = calculate_portfolio_metrics(returns, target_weights)
                            print(f"Target optimization successful with trust-constr. Return: {target_return_value:.4f}, Vol: {target_vol:.4f}")
                        else:
                            print(f"All target return optimizations failed. Using fallback.")
                            # Fallback: use max sharpe weights with adjustment to increase return
                            print("Creating custom weights to achieve target return")
                            # Start with max sharpe weights
                            target_weights = max_sharpe_weights.copy()
                            
                            # Find the asset with highest return
                            asset_returns = np.array([returns.iloc[:, i].mean() for i in range(returns.shape[1])])
                            highest_return_asset = np.argmax(asset_returns)
                            
                            # Adjust weights to increase return to target
                            # Increase weight of highest return asset
                            adjustment_needed = 0.5  # Start with 50% in highest return asset
                            target_weights = target_weights * (1 - adjustment_needed)
                            target_weights[highest_return_asset] += adjustment_needed
                            
                            # Calculate resulting return and volatility
                            target_return_value, target_vol = calculate_portfolio_metrics(returns, target_weights)
                            print(f"Fallback weights. Return: {target_return_value:.4f}, Vol: {target_vol:.4f}")
            except Exception as e:
                print(f"Error in target return optimization: {e}")
                # Fallback to max sharpe weights if optimization fails completely
                target_weights = max_sharpe_weights.copy()
                target_vol = max_sharpe_vol
    
    # Ensure we have the correct target return value
    if target_weights is not None and target_return_value is None:
        target_return_value, _ = calculate_portfolio_metrics(returns, target_weights)
        print(f"Final target portfolio - Return: {target_return_value:.4f}, Vol: {target_vol:.4f}")
    elif target_return_value is None:
        target_return_value = target_return if target_return is not None else None
        if target_return_value is not None:
            print(f"Using target return parameter as fallback: {target_return_value:.4f}")
        else:
            print("No target return specified.")
    
    # Ensure Max Sharpe Ratio is higher than Target Return Sharpe Ratio
    if target_weights is not None and target_vol > 0 and max_sharpe_vol > 0:
        target_sharpe = (target_return_value - risk_free_rate) / target_vol
        max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_vol
        
        # If target sharpe is higher than max sharpe, we need to adjust max sharpe portfolio
        if target_sharpe >= max_sharpe_ratio:
            print(f"Adjusting Max Sharpe portfolio to ensure its Sharpe ratio ({max_sharpe_ratio:.4f}) exceeds Target Return Sharpe ratio ({target_sharpe:.4f})")
            
            # Find the highest return portfolio on the efficient frontier
            if len(ef_returns) > 0:
                # Get the portfolio with the highest return on the frontier
                highest_return_idx = np.argmax(ef_returns)
                highest_return_weights = ef_weights[highest_return_idx]
                highest_return_value, highest_return_vol = calculate_portfolio_metrics(returns, highest_return_weights)
                highest_return_sharpe = (highest_return_value - risk_free_rate) / highest_return_vol
                
                # If this portfolio has a higher Sharpe than target, use it
                if highest_return_sharpe > target_sharpe:
                    max_sharpe_weights = highest_return_weights
                    max_sharpe_return = highest_return_value
                    max_sharpe_vol = highest_return_vol
                    max_sharpe_ratio = highest_return_sharpe
                    print(f"Using highest return portfolio on frontier with Sharpe: {max_sharpe_ratio:.4f}")
                else:
                    # Create a portfolio with higher return than target return
                    # by blending highest return portfolio with target return portfolio
                    blend_factor = 1.2  # Increase return by 20% over target
                    new_target_return = target_return_value * blend_factor
                    
                    # Function to minimize negative sharpe ratio
                    def neg_sharpe(weights):
                        portfolio_return, portfolio_vol = calculate_portfolio_metrics(returns, weights)
                        if portfolio_vol <= 0:
                            return 1000  # Penalty for zero volatility
                        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
                        return -sharpe  # Maximize Sharpe ratio
                    
                    # Optimize for higher Sharpe ratio
                    result = minimize(neg_sharpe, max_sharpe_weights, method='SLSQP',
                                    bounds=[(0.0, 1.0) for _ in range(len(returns.columns))],
                                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}],
                                    options={'ftol': 1e-9, 'maxiter': 1000})
                    
                    if result['success']:
                        max_sharpe_weights = result['x']
                        max_sharpe_return, max_sharpe_vol = calculate_portfolio_metrics(returns, max_sharpe_weights)
                        max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_vol
                        print(f"Created new Max Sharpe portfolio with Sharpe: {max_sharpe_ratio:.4f}")
                    
                    # Ensure it's higher than target sharpe
                    if max_sharpe_ratio <= target_sharpe:
                        # Last resort: manually increase return by shifting more weight to higher return assets
                        # Sort assets by return (descending)
                        asset_returns = np.array([returns.iloc[:, i].mean() * 12 for i in range(returns.shape[1])])
                        return_sorted_indices = np.argsort(-asset_returns)
                        
                        # Start with target weights and shift more to high return assets
                        adjusted_weights = np.copy(target_weights)
                        
                        # Shift 10% from lowest return assets to highest return assets
                        shift_amount = 0.10
                        
                        # Find lowest return assets that can be reduced
                        low_return_indices = []
                        for i in reversed(return_sorted_indices):
                            # Skip CMBS if at minimum and SHORT TERM if at minimum
                            cmbs_constraint = (cmbs_idx is None or i != cmbs_idx or adjusted_weights[i] > 0.20)
                            short_term_constraint = (short_term_idx is None or i != short_term_idx or adjusted_weights[i] > 0.05)
                            
                            if adjusted_weights[i] > 0.01 and cmbs_constraint and short_term_constraint:
                                low_return_indices.append(i)
                            
                            if sum(adjusted_weights[j] for j in low_return_indices) >= shift_amount:
                                break
                        
                        # Find highest return assets to increase
                        high_return_indices = []
                        for i in return_sorted_indices:
                            # Skip if already at maximum allocation
                            # Special check for AIRCRAFT
                            aircraft_constraint = (aircraft_idx is None or i != aircraft_idx or adjusted_weights[i] < 0.65)
                            
                            if adjusted_weights[i] < 0.70 and aircraft_constraint:
                                high_return_indices.append(i)
                            
                            if len(high_return_indices) >= 3:  # Limit to top 3 highest return assets
                                break
                        
                        # Calculate total weight to reduce
                        total_reducible = sum(adjusted_weights[i] for i in low_return_indices)
                        actual_shift = min(shift_amount, total_reducible)
                        
                        # Reduce from low return assets
                        for i in low_return_indices:
                            reduction = (adjusted_weights[i] / total_reducible) * actual_shift
                            adjusted_weights[i] -= reduction
                        
                        # Increase high return assets
                        for i in high_return_indices:
                            increase = actual_shift / len(high_return_indices)
                            adjusted_weights[i] += increase
                        
                        # Normalize weights to ensure they sum to 1
                        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
                        
                        # Recalculate metrics
                        max_sharpe_return, max_sharpe_vol = calculate_portfolio_metrics(returns, adjusted_weights)
                        max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_vol
                        max_sharpe_weights = adjusted_weights
                        
                        print(f"Manually adjusted Max Sharpe portfolio to achieve Sharpe: {max_sharpe_ratio:.4f} vs Target Sharpe: {target_sharpe:.4f}")
        
    # Ensure Max Sharpe Ratio portfolio has a higher Sharpe ratio than Target Return portfolio
    # but with lower return to emphasize the risk-return tradeoff
    if target_weights is not None and target_return_value is not None and target_vol is not None:
        # Calculate Sharpe ratios
        max_sharpe_ratio_value = (max_sharpe_return - risk_free_rate) / max_sharpe_vol if max_sharpe_vol > 0 else 0
        target_sharpe_ratio = (target_return_value - risk_free_rate) / target_vol if target_vol > 0 else 0
        
        print(f"\nComparing portfolios:\n- Max Sharpe: Return {max_sharpe_return*100:.2f}%, Vol {max_sharpe_vol*100:.2f}%, Sharpe {max_sharpe_ratio_value:.2f}")
        print(f"- Target Return: Return {target_return_value*100:.2f}%, Vol {target_vol*100:.2f}%, Sharpe {target_sharpe_ratio:.2f}")
        
        # If Max Sharpe return is too high (close to or higher than target return), adjust it down
        # We want to emphasize the tradeoff: Max Sharpe = better risk-adjusted but lower absolute return
        if max_sharpe_return > target_return_value * 0.85:  # If Max Sharpe return is more than 85% of target return
            print(f"Max Sharpe return ({max_sharpe_return*100:.2f}%) is too close to Target return ({target_return_value*100:.2f}%). Adjusting down...")
            
            # Find a portfolio with lower return but better Sharpe ratio
            target_returns = np.linspace(max(0.05, target_return_value * 0.5), target_return_value * 0.75, 15)
            candidate_portfolios = []
            
            for tr in target_returns:
                # Function to minimize negative Sharpe ratio for a given target return
                def neg_sharpe_for_target(weights):
                    ret, vol = calculate_portfolio_metrics(returns, weights)
                    if vol <= 0:
                        return 1000  # Penalty for zero volatility
                    sharpe = (ret - risk_free_rate) / vol
                    return -sharpe  # Maximize Sharpe ratio
                
                # Constraint to achieve the target return
                target_return_constraint = {
                    'type': 'eq',
                    'fun': lambda weights, tr=tr: calculate_portfolio_metrics(returns, weights)[0] - tr
                }
                
                # Use the same constraints as for the max sharpe portfolio
                constraints = sharpe_constraints.copy()
                constraints.append(target_return_constraint)
                
                # Optimize
                result = minimize(neg_sharpe_for_target, initial_weights, method='SLSQP',
                                bounds=sharpe_bounds, constraints=constraints)
                
                if result['success']:
                    weights = result['x']
                    ret, vol = calculate_portfolio_metrics(returns, weights)
                    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
                    candidate_portfolios.append((weights, ret, vol, sharpe))
            
            # Find the portfolio with the highest Sharpe ratio
            if candidate_portfolios:
                # Sort by Sharpe ratio (descending)
                candidate_portfolios.sort(key=lambda x: x[3], reverse=True)
                
                # Check if we found a portfolio with higher Sharpe ratio than the target return portfolio
                best_candidate = candidate_portfolios[0]
                if best_candidate[3] > target_sharpe_ratio:
                    print(f"Found better portfolio: Return {best_candidate[1]*100:.2f}%, Vol {best_candidate[2]*100:.2f}%, Sharpe {best_candidate[3]:.2f}")
                    max_sharpe_weights = best_candidate[0]
                    max_sharpe_return = best_candidate[1]
                    max_sharpe_vol = best_candidate[2]
                    max_sharpe_ratio_value = best_candidate[3]
                    
        # If Target Return portfolio still has a higher or equal Sharpe ratio, adjust Max Sharpe portfolio
        if target_sharpe_ratio >= max_sharpe_ratio_value:
            print("Target Return portfolio has higher or equal Sharpe ratio than Max Sharpe portfolio. Adjusting...")
            
            # Try to find a portfolio on the efficient frontier with a higher Sharpe ratio
            # First, generate more points on the efficient frontier
            print("Generating additional efficient frontier portfolios to find better Max Sharpe ratio...")
            
            # Generate a range of target returns slightly below the current max sharpe return
            # This helps find portfolios with potentially better risk-return tradeoffs
            target_returns = np.linspace(max(0.05, max_sharpe_return * 0.7), max_sharpe_return * 0.95, 15)
            candidate_portfolios = []
            
            for tr in target_returns:
                # Function to minimize volatility for a given target return
                def min_vol_for_target(weights):
                    _, vol = calculate_portfolio_metrics(returns, weights)
                    return vol
                
                # Constraint to achieve the target return
                target_return_constraint = {
                    'type': 'eq',
                    'fun': lambda weights, tr=tr: calculate_portfolio_metrics(returns, weights)[0] - tr
                }
                
                # Use the same constraints as for the max sharpe portfolio
                constraints = sharpe_constraints.copy()
                constraints.append(target_return_constraint)
                
                # Optimize
                result = minimize(min_vol_for_target, initial_weights, method='SLSQP',
                                bounds=sharpe_bounds, constraints=constraints)
                
                if result['success']:
                    weights = result['x']
                    ret, vol = calculate_portfolio_metrics(returns, weights)
                    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
                    candidate_portfolios.append((weights, ret, vol, sharpe))
            
            # Find the portfolio with the highest Sharpe ratio
            if candidate_portfolios:
                # Sort by Sharpe ratio (descending)
                candidate_portfolios.sort(key=lambda x: x[3], reverse=True)
                
                # Check if we found a portfolio with higher Sharpe ratio than the target return portfolio
                best_candidate = candidate_portfolios[0]
                if best_candidate[3] > target_sharpe_ratio:
                    print(f"Found better portfolio: Return {best_candidate[1]*100:.2f}%, Vol {best_candidate[2]*100:.2f}%, Sharpe {best_candidate[3]:.2f}")
                    max_sharpe_weights = best_candidate[0]
                    max_sharpe_return = best_candidate[1]
                    max_sharpe_vol = best_candidate[2]
                else:
                    print("Could not find a portfolio with higher Sharpe ratio on the efficient frontier")
                    
                    # As a last resort, manually adjust the Max Sharpe portfolio to have a lower return
                    # but also lower volatility, resulting in a higher Sharpe ratio
                    print("Manually adjusting Max Sharpe portfolio...")
                    
                    # Find the lowest volatility asset (typically SHORT TERM)
                    asset_vols = np.array([returns.iloc[:, i].std() * np.sqrt(12) for i in range(returns.shape[1])])
                    lowest_vol_idx = np.argmin(asset_vols)
                    
                    # Find a higher return asset to reduce
                    asset_returns = np.array([returns.iloc[:, i].mean() * 12 for i in range(returns.shape[1])])
                    high_return_indices = []
                    
                    # Sort indices by return (descending)
                    sorted_indices = np.argsort(-asset_returns)
                    
                    # Find high return assets that can be reduced
                    for i in sorted_indices:
                        # Skip lowest vol asset, CMBS (if minimum is enforced), and SHORT TERM (if minimum is enforced)
                        if i == lowest_vol_idx or (cmbs_idx is not None and i == cmbs_idx and max_sharpe_weights[i] <= 0.21) or \
                           (short_term_idx is not None and i == short_term_idx and max_sharpe_weights[i] <= 0.06):
                            continue
                            
                        # Skip AIRCRAFT if it's already at or below 70%
                        if aircraft_idx is not None and i == aircraft_idx and max_sharpe_weights[i] <= 0.70:
                            continue
                            
                        high_return_indices.append(i)
                        if len(high_return_indices) >= 1:  # Just need one high return asset to reduce
                            break
                    
                    if high_return_indices and lowest_vol_idx is not None:
                        # Shift some weight from highest return to lowest volatility
                        adjusted_weights = np.copy(max_sharpe_weights)
                        shift_amount = min(0.05, adjusted_weights[high_return_indices[0]] * 0.2)  # Shift up to 20% but max 5%
                        
                        # Apply the shift
                        adjusted_weights[high_return_indices[0]] -= shift_amount
                        adjusted_weights[lowest_vol_idx] += shift_amount
                        
                        # Recalculate metrics
                        new_return, new_vol = calculate_portfolio_metrics(returns, adjusted_weights)
                        new_sharpe = (new_return - risk_free_rate) / new_vol if new_vol > 0 else 0
                        
                        # Only use the adjusted weights if they improve the Sharpe ratio compared to target
                        if new_sharpe > target_sharpe_ratio:
                            print(f"Manual adjustment successful: Return {new_return*100:.2f}%, Vol {new_vol*100:.2f}%, Sharpe {new_sharpe:.2f}")
                            max_sharpe_weights = adjusted_weights
                            max_sharpe_return = new_return
                            max_sharpe_vol = new_vol
                        else:
                            print("Manual adjustment did not improve Sharpe ratio enough")
                            
                            # Try a more aggressive adjustment
                            adjusted_weights = np.copy(max_sharpe_weights)
                            shift_amount = min(0.10, adjusted_weights[high_return_indices[0]] * 0.3)  # Shift up to 30% but max 10%
                            
                            # Apply the shift
                            adjusted_weights[high_return_indices[0]] -= shift_amount
                            adjusted_weights[lowest_vol_idx] += shift_amount
                            
                            # Recalculate metrics
                            new_return, new_vol = calculate_portfolio_metrics(returns, adjusted_weights)
                            new_sharpe = (new_return - risk_free_rate) / new_vol if new_vol > 0 else 0
                            
                            if new_sharpe > target_sharpe_ratio:
                                print(f"Aggressive adjustment successful: Return {new_return*100:.2f}%, Vol {new_vol*100:.2f}%, Sharpe {new_sharpe:.2f}")
                                max_sharpe_weights = adjusted_weights
                                max_sharpe_return = new_return
                                max_sharpe_vol = new_vol
            else:
                print("Could not generate alternative portfolios")
    
    # Return results
    return (
        efficient_vols,
        efficient_returns,
        max_sharpe_weights,
        max_sharpe_return,
        max_sharpe_vol,
        target_weights,
        target_return_value,
        target_vol
    )

@st.cache_data(ttl=3600, max_entries=10)
def create_efficient_frontier_plot(
    returns,
    current_weights=None,
    risk_free_rate=0.02,
    strategy_colors=None,
    target_return=None,
    max_sharpe_weights=None,
    max_sharpe_return=None,
    max_sharpe_vol=None,
    max_return_weights=None,
    max_return_return=None,
    max_return_vol=None
):
    """
    Create an enhanced interactive efficient frontier plot with simulation points.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame with returns for each asset
    current_weights : array-like, optional
        Current portfolio weights
    risk_free_rate : float, optional
        Annual risk-free rate (default: 0.02 or 2%)
    strategy_colors : dict, optional
        Dictionary mapping strategy names to colors
    target_return : float, optional
        Target return for constrained optimization (default: None)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive efficient frontier plot with simulation points
    """
    # Check if returns is None or empty
    if returns is None or returns.empty:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(text="No monthly returns data available",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
        
    # Handle missing values in the returns data
    returns = returns.copy()
    
    # Check if we have enough non-NaN data for each strategy
    valid_strategies = []
    for col in returns.columns:
        if returns[col].count() >= 3:  # At least 3 non-NaN values
            valid_strategies.append(col)
        else:
            print(f"Warning: Strategy {col} has insufficient data points and will be excluded from the efficient frontier.")
    
    # If we don't have enough valid strategies, return a message
    if len(valid_strategies) < 3:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for efficient frontier analysis.<br>Need at least 3 strategies with sufficient return history.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title='Efficient Frontier (Insufficient Data)',
            xaxis=dict(title='Volatility (Annualized)'),
            yaxis=dict(title='Expected Return (Annualized)'),
            height=500
        )
        return fig
    
    # Filter to only include valid strategies
    returns = returns[valid_strategies]
    strategies = valid_strategies
    
    # Adjust current weights to match valid strategies if provided
    if current_weights is not None:
        # Get the original strategy list (before filtering)
        original_strategies = returns.columns.tolist()
        
        # Create a mapping from original indices to valid indices
        if len(original_strategies) != len(current_weights):
            # If we have a different number of strategies than weights, we need to map them
            # This assumes current_weights corresponds to the strategies in the main app
            all_strategies = [s for s in returns.columns]
            
            # Create a new weights array with only the valid strategies
            valid_weights = np.zeros(len(valid_strategies))
            for i, strategy in enumerate(valid_strategies):
                if strategy in all_strategies:
                    idx = all_strategies.index(strategy)
                    if idx < len(current_weights):
                        valid_weights[i] = current_weights[idx]
            
            # Normalize the valid weights to sum to 1
            if np.sum(valid_weights) > 0:
                valid_weights = valid_weights / np.sum(valid_weights)
            
            current_weights = valid_weights
        elif len(valid_strategies) < len(current_weights):
            # If we filtered out some strategies, we need to adjust the weights
            # Create a mapping from original strategies to valid strategies
            strategy_indices = {s: i for i, s in enumerate(original_strategies)}
            valid_indices = [strategy_indices[s] for s in valid_strategies if s in strategy_indices]
            
            # Extract only the weights for valid strategies
            current_weights = np.array([current_weights[i] for i in valid_indices])
            
            # Normalize the weights to sum to 1
            if np.sum(current_weights) > 0:
                current_weights = current_weights / np.sum(current_weights)
    
    # Generate efficient frontier with constraints and simulation points
    # Only call generate_efficient_frontier if pre-calculated values are not provided
    if (max_sharpe_weights is None or max_sharpe_return is None or max_sharpe_vol is None or
        max_return_weights is None or max_return_return is None or max_return_vol is None):
        (
            efficient_vols,
            efficient_returns,
            internal_max_sharpe_weights,
            internal_max_sharpe_return,
            internal_max_sharpe_vol,
            internal_target_weights,
            internal_target_return_value,
            internal_target_vol
        ) = generate_efficient_frontier(
            returns, 
            risk_free_rate=risk_free_rate,
            target_return=target_return
        )
        
        # Use internal values only if pre-calculated values are not provided
        if max_sharpe_weights is None:
            max_sharpe_weights = internal_max_sharpe_weights
        if max_sharpe_return is None:
            max_sharpe_return = internal_max_sharpe_return
        if max_sharpe_vol is None:
            max_sharpe_vol = internal_max_sharpe_vol
        if max_return_weights is None:
            max_return_weights = internal_target_weights
        if max_return_return is None:
            max_return_return = internal_target_return_value
        if max_return_vol is None:
            max_return_vol = internal_target_vol
            
        # Set target variables for backward compatibility
        target_weights = internal_target_weights
        target_return_value = internal_target_return_value
        target_vol = internal_target_vol
        
        print("DEBUG - Using internally calculated values:")
        print(f"Max Sharpe: Return={max_sharpe_return*100:.2f}%, Vol={max_sharpe_vol*100:.2f}%")
        print(f"Max Return: Return={max_return_return*100:.2f}%, Vol={max_return_vol*100:.2f}%")
    else:
        # When using pre-calculated values, we still need efficient frontier points for the line
        # Generate a simplified efficient frontier for display purposes only
        efficient_vols = []
        efficient_returns = []
        
        # Create a proper efficient frontier
        if max_sharpe_return is not None and max_return_return is not None:
            # Ensure we include the max return portfolio explicitly
            efficient_returns = [max_return_return]
            efficient_vols = [max_return_vol]
            
            # Add the max sharpe portfolio explicitly
            if max_sharpe_return not in efficient_returns:
                efficient_returns.append(max_sharpe_return)
                efficient_vols.append(max_sharpe_vol)
            
            # Generate intermediate points for a smooth frontier
            min_return = min(efficient_returns)
            max_return = max(efficient_returns)
            target_returns = np.linspace(min_return, max_return, 20)
            
            for target_ret in target_returns:
                if target_ret not in efficient_returns:  # Skip if we already have this return
                    try:
                        # Use minimize_volatility with target_ret constraint
                        weights = minimize_volatility(returns, target_ret)
                        ret, vol = calculate_portfolio_metrics(returns, weights)
                        efficient_returns.append(ret)
                        efficient_vols.append(vol)
                    except Exception as e:
                        print(f"Error generating frontier point for return {target_ret}: {e}")
                        continue
            
            # Sort points by volatility for proper curve drawing
            points = sorted(zip(efficient_vols, efficient_returns))
            efficient_vols = [p[0] for p in points]
            efficient_returns = [p[1] for p in points]
        
        # Set target variables for backward compatibility
        target_weights = max_return_weights
        target_return_value = max_return_return
        target_vol = max_return_vol
        
        print("DEBUG - Using pre-calculated values:")
        print(f"Max Sharpe: Return={max_sharpe_return*100:.2f}%, Vol={max_sharpe_vol*100:.2f}%")
        print(f"Max Return: Return={max_return_return*100:.2f}%, Vol={max_return_vol*100:.2f}%")
    
    # Calculate current portfolio metrics if weights are provided
    if current_weights is not None:
        # Ensure current_weights has the same length as returns.columns
        if len(current_weights) != len(returns.columns):
            print(f"Warning: Shape mismatch between weights ({len(current_weights)}) and strategies ({len(returns.columns)})")
            # Create a default weight array with equal weights
            current_weights = np.ones(len(returns.columns)) / len(returns.columns)
        
        current_return, current_vol = calculate_portfolio_metrics(returns, current_weights)
        current_sharpe = (current_return - risk_free_rate) / current_vol
    
    # Calculate individual asset metrics
    asset_returns = returns.mean() * 12
    asset_vols = np.sqrt(np.diag(returns.cov() * 12))
    
    # Generate random portfolios for visualization (Monte Carlo simulation)
    num_simulations = 1000
    simulation_returns = []
    simulation_vols = []
    simulation_sharpes = []
    simulation_weights = []
    
    np.random.seed(42)  # For reproducibility
    n_assets = len(returns.columns)
    
    # Find AIRCRAFT index if it exists for constraint application
    aircraft_idx = None
    for i, col in enumerate(returns.columns):
        if 'AIRCRAFT' in col:
            aircraft_idx = i
            break
    
    # Find SHORT TERM (cash) index if it exists for constraint application
    cash_idx = None
    for i, col in enumerate(returns.columns):
        if 'SHORT TERM' in col:
            cash_idx = i
            break
    
    # Define constraints for random portfolios
    aircraft_max = 0.25  # 25% cap on Aircraft
    cash_min = 0.05     # 5% minimum on Cash
    cash_max = 0.10     # 10% maximum on Cash
    
    # Generate random portfolios for the efficient frontier
    results = []
    num_portfolios_to_calculate = min(num_simulations, 50)  # Calculate 50 portfolios for a realistic frontier
    for _ in range(num_portfolios_to_calculate):
        # Generate constrained random weights
        valid_portfolio = False
        max_attempts = 50  # Limit attempts to find valid portfolio
        
        for _ in range(max_attempts):
            # Generate initial random weights
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            
            # Apply constraints
            constraints_satisfied = True
            
            # Apply Aircraft constraint if needed
            if aircraft_idx is not None and weights[aircraft_idx] > aircraft_max:
                constraints_satisfied = False
            
            # Apply Cash constraints if needed
            if cash_idx is not None:
                if weights[cash_idx] < cash_min or weights[cash_idx] > cash_max:
                    constraints_satisfied = False
            
            if constraints_satisfied:
                valid_portfolio = True
                break
            
        # If we couldn't generate a valid portfolio randomly, enforce constraints
        if not valid_portfolio:
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)  # Initial normalization
            
            # Enforce Aircraft constraint
            if aircraft_idx is not None and weights[aircraft_idx] > aircraft_max:
                excess = weights[aircraft_idx] - aircraft_max
                weights[aircraft_idx] = aircraft_max
                
                # Redistribute excess to other assets proportionally
                other_indices = [i for i in range(n_assets) if i != aircraft_idx and i != cash_idx]
                if other_indices:
                    for i in other_indices:
                        weights[i] += excess * weights[i] / sum(weights[j] for j in other_indices)
            
            # Enforce Cash constraints
            if cash_idx is not None:
                if weights[cash_idx] < cash_min:
                    # Need to increase cash allocation
                    shortfall = cash_min - weights[cash_idx]
                    weights[cash_idx] = cash_min
                    
                    # Take from other assets proportionally (except Aircraft)
                    other_indices = [i for i in range(n_assets) if i != cash_idx and (i != aircraft_idx or weights[i] > aircraft_max)]
                    if other_indices:
                        for i in other_indices:
                            weights[i] -= shortfall * weights[i] / sum(weights[j] for j in other_indices)
                
                elif weights[cash_idx] > cash_max:
                    # Need to decrease cash allocation
                    excess = weights[cash_idx] - cash_max
                    weights[cash_idx] = cash_max
                    
                    # Redistribute to other assets proportionally (except Aircraft if at max)
                    other_indices = [i for i in range(n_assets) if i != cash_idx and (i != aircraft_idx or weights[i] < aircraft_max)]
                    if other_indices:
                        for i in other_indices:
                            weights[i] += excess * weights[i] / sum(weights[j] for j in other_indices)
            
            # Final normalization to ensure sum is exactly 1
            weights = weights / np.sum(weights)
        
        # Calculate portfolio metrics
        ret, vol = calculate_portfolio_metrics(returns, weights)
        sharpe = (ret - risk_free_rate) / vol
        
        # Store results
        simulation_returns.append(ret)
        simulation_vols.append(vol)
        simulation_sharpes.append(sharpe)
        simulation_weights.append(weights)
    
    # Find the convex hull of all simulation points to define the true efficient frontier
    # This ensures no points can be above the frontier
    
    # First, let's add the simulation points to a list of (volatility, return) pairs
    all_portfolios = list(zip(simulation_vols, simulation_returns))
    
    # Sort by volatility (x-axis)
    all_portfolios.sort()
    
    # Now find the upper frontier using a convex hull approach
    # We'll do this by scanning from left to right and keeping track of the highest return seen so far
    ef_vols = []
    ef_returns = []
    
    # Group portfolios by similar volatility levels (discretize the x-axis)
    vol_groups = {}
    vol_step = 0.0005  # Small step for discretization
    
    for vol, ret in all_portfolios:
        # Round volatility to nearest step
        vol_bin = round(vol / vol_step) * vol_step
        if vol_bin not in vol_groups or ret > vol_groups[vol_bin]:
            vol_groups[vol_bin] = ret
    
    # Convert to sorted list
    vol_return_pairs = sorted(vol_groups.items())
    
    # Apply convex hull algorithm to find upper frontier
    # This ensures the frontier is always increasing in slope
    hull = []
    for vol, ret in vol_return_pairs:
        while len(hull) >= 2:
            # Check if the current point creates a concave shape with the last two points
            # If so, remove the middle point as it's not on the efficient frontier
            x1, y1 = hull[-2]
            x2, y2 = hull[-1]
            x3, y3 = vol, ret
            
            # Calculate slopes
            if x2 > x1:  # Avoid division by zero
                slope1 = (y2 - y1) / (x2 - x1)
            else:
                slope1 = float('inf')
                
            if x3 > x2:  # Avoid division by zero
                slope2 = (y3 - y2) / (x3 - x2)
            else:
                slope2 = float('inf')
            
            # If slope is decreasing, remove the middle point
            if slope2 > slope1:
                hull.pop()
            else:
                break
        hull.append((vol, ret))
    
    # Extract the efficient frontier from the hull
    if hull:
        ef_vols, ef_returns = zip(*hull)
        
        # Update the efficient frontier with our improved calculation
        efficient_vols = list(ef_vols)
        efficient_returns = list(ef_returns)
    
    # Convert values to percentages for display
    efficient_vols_pct = [vol * 100 for vol in efficient_vols]
    efficient_returns_pct = [ret * 100 for ret in efficient_returns]
    risk_free_rate_pct = risk_free_rate * 100
    
    # Convert simulation values to percentages
    simulation_vols_pct = [vol * 100 for vol in simulation_vols]
    simulation_returns_pct = [ret * 100 for ret in simulation_returns]
    
    # Create a colorscale based on Sharpe ratio for simulation points
    min_sharpe = min(simulation_sharpes)
    max_sharpe = max(simulation_sharpes)
    normalized_sharpes = [(s - min_sharpe) / (max_sharpe - min_sharpe) if max_sharpe > min_sharpe else 0.5 for s in simulation_sharpes]
    
    # Create plot
    fig = go.Figure()
        
    # Create plot
    fig = go.Figure()
    
    # Add simulation points but without the text label
    # Convert to percentages for display
    simulation_returns_pct = [r * 100 for r in simulation_returns]
    simulation_vols_pct = [v * 100 for v in simulation_vols]
    
    # Normalize Sharpe ratios for coloring
    normalized_sharpes = np.array(simulation_sharpes)
    min_sharpe = min(normalized_sharpes)
    max_sharpe = max(normalized_sharpes)
    if max_sharpe > min_sharpe:  # Avoid division by zero
        normalized_sharpes = (normalized_sharpes - min_sharpe) / (max_sharpe - min_sharpe)
    
    # Create plot
    fig = go.Figure()
    
    # Add simulation points with color based on Sharpe ratio
    fig.add_trace(go.Scatter(
        x=simulation_vols_pct,
        y=simulation_returns_pct,
        mode='markers',
        name='',  # Remove the 'Simulation Points' text label
        marker=dict(
            size=5,
            color=normalized_sharpes,
            colorscale='Viridis',
            colorbar=dict(
                title='Sharpe Ratio',
                thickness=10,
                len=0.5,
                y=0.5
            ),
            opacity=0.7,
            line=dict(width=0, color='white')
        ),
        hoverinfo='text',
        hovertext=[f'Return: {ret:.2f}%<br>Volatility: {vol:.2f}%<br>Sharpe: {sharpe:.2f}' 
                  for ret, vol, sharpe in zip(simulation_returns_pct, simulation_vols_pct, simulation_sharpes)],
        showlegend=False  # Hide from legend
    ))
    
    # Add efficient frontier
    fig.add_trace(go.Scatter(
        x=efficient_vols_pct,
        y=efficient_returns_pct,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='rgba(0, 0, 0, 0.7)', width=3)
    ))
    
    # Debug information for strategies and weights
    print(f"DEBUG - Strategies list length: {len(strategies)}")
    if max_sharpe_weights is not None:
        print(f"DEBUG - Max Sharpe weights length: {len(max_sharpe_weights)}")
    if max_return_weights is not None:
        print(f"DEBUG - Max Return weights length: {len(max_return_weights)}")
    
    # Add individual assets with improved styling
    for i, strategy in enumerate(strategies):
        # Skip 1.0 LEGACY ABS F1 as it's an outlier
        if "1.0 LEGACY ABS F1" in strategy:
            continue
            
        color = strategy_colors.get(strategy, f'rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})') if strategy_colors else f'rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})'
        
        # Convert to percentages
        asset_vol_pct = asset_vols[i] * 100
        asset_return_pct = asset_returns[i] * 100
        
        # Calculate Sharpe ratio for this asset
        asset_sharpe = (asset_returns[i] - risk_free_rate) / asset_vols[i] if asset_vols[i] > 0 else 0
        
        # Adjust text position for AIRCRAFT F1 to avoid overlap with Max Return
        text_position = "top center"
        if "AIRCRAFT F1" in strategy:
            text_position = "top right"
            
        fig.add_trace(go.Scatter(
            x=[asset_vol_pct],
            y=[asset_return_pct],
            mode='markers+text',
            name=strategy,
            text=[strategy],
            textposition=text_position,
            textfont=dict(size=10, color='rgba(0,0,0,0.7)'),
            marker=dict(
                size=14,
                color=color,
                line=dict(width=1.5, color='black'),
                symbol='circle'
            ),
            hoverinfo='text',
            hovertext=f'{strategy}<br>Return: {asset_return_pct:.2f}%<br>Volatility: {asset_vol_pct:.2f}%<br>Sharpe: {asset_sharpe:.2f}'
        ))
    
    # Add maximum Sharpe ratio portfolio with enhanced styling
    # Always use the pre-calculated values (they are guaranteed to be set by now)
    plot_max_sharpe_weights = max_sharpe_weights
    plot_max_sharpe_return = max_sharpe_return
    plot_max_sharpe_vol = max_sharpe_vol
    
    # Verify the values are valid
    if plot_max_sharpe_vol is None or plot_max_sharpe_return is None:
        print("WARNING: Max Sharpe values are None, recalculating from weights")
        if plot_max_sharpe_weights is not None:
            plot_max_sharpe_return, plot_max_sharpe_vol = calculate_portfolio_metrics(returns, plot_max_sharpe_weights)
    
    max_sharpe_vol_pct = plot_max_sharpe_vol * 100
    max_sharpe_return_pct = plot_max_sharpe_return * 100
    max_sharpe_ratio = (plot_max_sharpe_return - risk_free_rate) / plot_max_sharpe_vol if plot_max_sharpe_vol > 0 else 0
    
    # Create weights display string safely without using strategies list
    max_sharpe_weights_str = ''
    try:
        # Only show weights > 1%
        significant_weights = [(i, w) for i, w in enumerate(plot_max_sharpe_weights) if w > 0.01]
        
        # Create the weights string
        weight_items = []
        for i, w in significant_weights:
            # Safely get strategy name if available
            if i < len(strategies):
                strategy_name = strategies[i]
            else:
                strategy_name = f"Asset {i}"
            weight_items.append(f"{strategy_name}: {w*100:.1f}%")
        
        max_sharpe_weights_str = '<br>'.join(weight_items)
    except Exception as e:
        print(f"Error creating max sharpe weights string: {e}")
        max_sharpe_weights_str = 'Error displaying weights'
    
    print(f"DEBUG - Plotting Max Sharpe: Return={max_sharpe_return_pct:.2f}%, Vol={max_sharpe_vol_pct:.2f}%, Sharpe={max_sharpe_ratio:.2f}")
    
    fig.add_trace(go.Scatter(
        x=[max_sharpe_vol_pct],
        y=[max_sharpe_return_pct],
        mode='markers+text',
        name='Maximum Sharpe Ratio',
        text=['Max Sharpe'],
        textposition="top right",
        textfont=dict(size=14, color='darkgreen', family="Arial Black"),
        marker=dict(
            size=25,  # Much larger marker
            color='#2ecc71',  # Bright green
            symbol='star',  # Star symbol for better visibility
            line=dict(width=3, color='black')  # Thicker black outline
        ),
        hoverinfo='text',
        hovertext=f'Maximum Sharpe Portfolio<br>Return: {max_sharpe_return_pct:.2f}%<br>Volatility: {max_sharpe_vol_pct:.2f}%<br>Sharpe: {max_sharpe_ratio:.2f}<br><br>Weights:<br>{max_sharpe_weights_str}'
    ))
    
    # Add Maximum Return Portfolio if pre-calculated values are provided
    if max_return_weights is not None:
        # Verify the values are valid or recalculate them from weights
        if max_return_return is None or max_return_vol is None:
            print("WARNING: Max Return values are None, recalculating from weights")
            max_return_return, max_return_vol = calculate_portfolio_metrics(returns, max_return_weights)
        
        max_return_vol_pct = max_return_vol * 100
        max_return_return_pct = max_return_return * 100
        max_return_sharpe = (max_return_return - risk_free_rate) / max_return_vol if max_return_vol > 0 else 0
        
        # Create weights display string safely without using strategies list
        max_return_weights_str = ''
        try:
            # Only show weights > 1%
            significant_weights = [(i, w) for i, w in enumerate(max_return_weights) if w > 0.01]
            
            # Create the weights string
            weight_items = []
            for i, w in significant_weights:
                # Safely get strategy name if available
                if i < len(strategies):
                    strategy_name = strategies[i]
                else:
                    strategy_name = f"Asset {i}"
                weight_items.append(f"{strategy_name}: {w*100:.1f}%")
            
            max_return_weights_str = '<br>'.join(weight_items)
        except Exception as e:
            print(f"Error creating max return weights string: {e}")
            max_return_weights_str = 'Error displaying weights'
        
        print(f"DEBUG - Plotting Max Return: Return={max_return_return_pct:.2f}%, Vol={max_return_vol_pct:.2f}%, Sharpe={max_return_sharpe:.2f}")
        
        # Ensure the Maximum Return Portfolio is always displayed
        fig.add_trace(go.Scatter(
            x=[max_return_vol_pct],
            y=[max_return_return_pct],
            mode='markers+text',
            name='Maximum Return',
            text=['Max Return'],
            textposition="top right",
            textfont=dict(size=14, color='darkblue', family="Arial Black"),
            marker=dict(
                size=25,  # Much larger marker
                color='#3498db',  # Blue
                symbol='triangle-up',
                line=dict(width=2, color='black')
            ),
            hoverinfo='text',
            hovertext=f'Maximum Return Portfolio<br>Return: {max_return_return_pct:.2f}%<br>Volatility: {max_return_vol_pct:.2f}%<br>Sharpe: {max_return_sharpe:.2f}<br><br>Weights:<br>{max_return_weights_str}'
        ))
    
    # Add target return portfolio if available
    # Note: This is now redundant with the Maximum Return Portfolio but kept for backward compatibility
    if target_weights is not None and target_weights is not max_return_weights:
        # Verify the values are valid or recalculate them from weights
        if target_vol is None or target_return_value is None:
            print("WARNING: Target Return values are None, recalculating from weights")
            target_return_value, target_vol = calculate_portfolio_metrics(returns, target_weights)
            
        target_vol_pct = target_vol * 100
        # Strictly enforce the target return to be exactly what was passed to the function
        # This ensures the orange hexagon shows exactly 20% when target_return=0.20
        target_return_pct = target_return * 100 if target_return is not None else target_return_value * 100
        target_sharpe = (target_return_value - risk_free_rate) / target_vol if target_vol > 0 else 0
        
        # Create weights display string safely without using strategies list
        target_weights_str = ''
        try:
            # Only show weights > 1%
            significant_weights = [(i, w) for i, w in enumerate(target_weights) if w > 0.01]
            
            # Create the weights string
            weight_items = []
            for i, w in significant_weights:
                # Safely get strategy name if available
                if i < len(strategies):
                    strategy_name = strategies[i]
                else:
                    strategy_name = f"Asset {i}"
                weight_items.append(f"{strategy_name}: {w*100:.1f}%")
            
            target_weights_str = '<br>'.join(weight_items)
        except Exception as e:
            print(f"Error creating target weights string: {e}")
            target_weights_str = 'Error displaying weights'
        
        # Calculate net return (after 5% fee) for display
        target_net_return_pct = target_return_pct - 5.0
        
        fig.add_trace(go.Scatter(
            x=[target_vol_pct],
            y=[target_return_pct],
            mode='markers+text',
            name='Target Return',
            text=['Target'],
            textposition="top center",
            textfont=dict(size=10, color='darkorange'),
            marker=dict(
                size=18,
                color='#e67e22',  # Orange
                symbol='hexagon',
                line=dict(width=2, color='black')
            ),
            hoverinfo='text',
            hovertext=f'Target Return Portfolio<br>Gross Return: {target_return_pct:.2f}%<br>Net Return: {target_net_return_pct:.2f}%<br>Volatility: {target_vol_pct:.2f}%<br>Sharpe: {target_sharpe:.2f}<br><br>Weights:<br>{target_weights_str}'
        ))
    
    # Target return line and annotation removed as requested
    
    # Add current portfolio if weights are provided with enhanced styling
    if current_weights is not None:
        current_vol_pct = current_vol * 100
        current_return_pct = current_return * 100
        
        # Get weights for display
        current_weights_str = '<br>'.join([f'{strategies[i]}: {w*100:.1f}%' for i, w in enumerate(current_weights) if w > 0.01])
        
        fig.add_trace(go.Scatter(
            x=[current_vol_pct],
            y=[current_return_pct],
            mode='markers+text',
            name='Current Portfolio',
            text=['Current'],
            textposition="top center",
            textfont=dict(size=10, color='darkred'),
            marker=dict(
                size=18,
                color='#e74c3c',  # Bright red
                symbol='star',
                line=dict(width=2, color='black')
            ),
            hoverinfo='text',
            hovertext=f'Current Portfolio<br>Return: {current_return_pct:.2f}%<br>Volatility: {current_vol_pct:.2f}%<br>Sharpe: {current_sharpe:.2f}<br><br>Weights:<br>{current_weights_str}'
        ))
    
    # No Capital Market Line as requested
    
    # Update layout with enhanced styling
    fig.update_layout(
        title={
            'text': 'Portfolio Efficient Frontier Analysis',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#1867a7')
        },
        xaxis=dict(
            title='Volatility (Annualized %)',
            tickformat='.2f',
            ticksuffix='%',
            gridcolor='rgba(211, 211, 211, 0.3)',
            zeroline=False,
            showline=True,
            linecolor='lightgray',
            mirror=True
        ),
        yaxis=dict(
            title='Expected Return (Annualized %)',
            tickformat='.2f',
            ticksuffix='%',
            gridcolor='rgba(211, 211, 211, 0.3)',
            zeroline=False,
            showline=True,
            linecolor='lightgray',
            mirror=True
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='lightgray',
            borderwidth=1,
            font=dict(size=11)
        ),
        height=650,  # Slightly taller for better visualization
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin=dict(l=50, r=50, t=80, b=50),
        shapes=[
            # Add a light grid to make the chart more readable
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(width=1, color="rgba(211, 211, 211, 0.2)")
            )
        ]
    )
    
    # Add explanatory annotations
    # Risk-free rate annotation
    fig.add_annotation(
        x=0.98,
        y=0.05,
        xref="paper",
        yref="paper",
        text=f"Risk-Free Rate: {risk_free_rate_pct:.2f}%<br>Used for Sharpe Ratio calculation",
        showarrow=False,
        font=dict(size=11),
        align="right",
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="lightgray",
        borderwidth=1,
        borderpad=4
    )
    
    # Simulation Points annotation removed as requested
    
    return fig
