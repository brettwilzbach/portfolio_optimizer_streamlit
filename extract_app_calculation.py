import pandas as pd
import numpy as np
import os
from pathlib import Path

def extract_app_calculation():
    """
    Extract the exact calculation used in the app for the 11.97% return figure.
    """
    print("App Portfolio Return Calculation Extraction")
    print("=========================================\n")
    
    # Load the monthly RoA data
    monthly_file = Path(__file__).parent / "Aggregate Monthly RoA.xlsx"
    if not monthly_file.exists():
        print(f"Error: Monthly data file not found: {monthly_file}")
        return
        
    print(f"Loading monthly data from: {monthly_file}")
    monthly_data = pd.read_excel(monthly_file)
    
    # Print basic info about the data
    print(f"Data shape: {monthly_data.shape}")
    print(f"Columns: {', '.join(monthly_data.columns[:10])}...")
    
    # Check for Month column
    if 'Month' not in monthly_data.columns:
        print("Error: 'Month' column not found in the data.")
        return
    
    # Load the current portfolio data to get actual weights
    try:
        # Try to find the portfolio data file
        portfolio_files = list(Path(__file__).parent.glob("*Portfolio*.xlsx"))
        if not portfolio_files:
            print("No portfolio data file found. Cannot proceed.")
            return
            
        portfolio_file = portfolio_files[0]
        print(f"Loading portfolio data from: {portfolio_file}")
        portfolio_data = pd.read_excel(portfolio_file)
        
        # Print portfolio data columns
        print(f"Portfolio data columns: {portfolio_data.columns.tolist()}")
        
        # Try to extract strategy weights
        if 'Strategy' in portfolio_data.columns and 'Admin Net MV' in portfolio_data.columns:
            # Calculate weights based on Admin Net MV
            total_mv = portfolio_data['Admin Net MV'].sum()
            strategy_weights = portfolio_data.groupby('Strategy')['Admin Net MV'].sum() / total_mv
            
            # Convert to dictionary
            weights = strategy_weights.to_dict()
            print("\nExtracted actual portfolio weights:")
            for strategy, weight in weights.items():
                print(f"{strategy}: {weight:.4f} ({weight*100:.1f}%)")
        else:
            print("Could not find Strategy or Admin Net MV columns in portfolio data.")
            return
    except Exception as e:
        print(f"Error loading portfolio data: {e}")
        return
    
    # Map strategy names to match the format in the monthly data
    strategy_mapping = {
        'AIRCRAFT F1': 'AIRCRAFT F1',
        'CMBS F1': 'CMBS F1',
        'SHORT TERM': 'SHORT TERM F1',  # Note the F1 suffix in the data
        'CLO F1': 'CLO F1',
        'ABS F1': 'ABS F1'
    }
    
    # Reverse mapping for display
    reverse_mapping = {v: k for k, v in strategy_mapping.items()}
    
    # Check which strategies exist in the data
    available_strategies = [s for s in strategy_mapping.values() if s in monthly_data.columns]
    if not available_strategies:
        print("No matching strategy columns found in the data.")
        return
        
    print(f"\nFound {len(available_strategies)} strategies in monthly data: {', '.join(available_strategies)}")
    
    # Convert Month column to datetime
    try:
        monthly_data['Month'] = pd.to_datetime(monthly_data['Month'])
    except:
        print("Warning: Could not convert Month column to datetime.")
    
    # Set the Month as index
    monthly_data.set_index('Month', inplace=True)
    
    # Extract the relevant columns
    monthly_returns = monthly_data[available_strategies]
    
    # Convert to numeric values
    for col in available_strategies:
        if monthly_returns[col].dtype == 'object':
            # Handle percentage strings
            try:
                if monthly_returns[col].astype(str).str.contains('%').any():
                    monthly_returns[col] = monthly_returns[col].str.rstrip('%').astype(float) / 100.0
                else:
                    monthly_returns[col] = pd.to_numeric(monthly_returns[col], errors='coerce')
            except:
                monthly_returns[col] = pd.to_numeric(monthly_returns[col], errors='coerce')
    
    # Set SHORT TERM to use a fixed yield (4.2% annual)
    short_term_yield = 4.2
    if 'SHORT TERM F1' in monthly_returns.columns:
        monthly_returns['SHORT TERM F1'] = short_term_yield / 100.0 / 12
        print(f"Set SHORT TERM F1 to fixed monthly yield: {short_term_yield/100.0/12:.6f} ({short_term_yield}% annual)")
    
    # Create weights array in the same order as the strategies
    # Map the weights to the available strategies in the monthly data
    current_weights = []
    for strategy in available_strategies:
        # Try to find the strategy in the weights dictionary
        # First try direct match
        if strategy in weights:
            current_weights.append(weights[strategy])
        # Then try using the reverse mapping
        elif strategy in reverse_mapping and reverse_mapping[strategy] in weights:
            current_weights.append(weights[reverse_mapping[strategy]])
        # Otherwise use 0
        else:
            current_weights.append(0.0)
    
    current_weights = np.array(current_weights)
    
    # Normalize weights to sum to 1
    if current_weights.sum() > 0:
        current_weights = current_weights / current_weights.sum()
    
    print("\nUsing weights (mapped to monthly data strategies):")
    for s, w in zip(available_strategies, current_weights):
        print(f"{s}: {w:.4f} ({w*100:.1f}%)")
    
    # Handle missing values
    returns_filled = monthly_returns.copy()
    for col in returns_filled.columns:
        returns_filled[col] = returns_filled[col].fillna(returns_filled[col].mean())
    
    # Calculate expected returns (annualized)
    expected_returns = returns_filled.mean() * 12
    
    # Print the detailed calculation (ARITHMETIC MEAN - APP METHOD)
    print("\n=== ARITHMETIC MEAN CALCULATION (APP METHOD) ===")
    print("Strategy | Monthly Mean | Annualized | Weight | Contribution")
    print("-" * 75)
    
    total_contribution = 0
    for i, strategy in enumerate(available_strategies):
        monthly_mean = returns_filled[strategy].mean()
        annualized = monthly_mean * 12
        weight = current_weights[i]
        contribution = annualized * weight
        total_contribution += contribution
        print(f"{strategy} | {monthly_mean:.6f} | {annualized:.6f} ({annualized*100:.2f}%) | {weight:.4f} ({weight*100:.1f}%) | {contribution:.6f} ({contribution*100:.2f}%)")
    
    print("-" * 75)
    print(f"Total portfolio return: {total_contribution:.6f} ({total_contribution*100:.2f}%)")
    
    # Calculate net return (after 5% fee)
    net_return = total_contribution - 0.05
    print(f"Net return (after 5% fee): {net_return:.6f} ({net_return*100:.2f}%)")
    
    # Calculate geometric mean return
    try:
        # Calculate geometric mean return
        # First convert monthly returns to 1+r format
        returns_plus_one = returns_filled + 1
        
        # Calculate portfolio return for each month
        monthly_portfolio_returns = np.zeros(len(returns_plus_one))
        for i in range(len(returns_plus_one)):
            # For each month, calculate the weighted return
            month_return = 0
            for j, strategy in enumerate(available_strategies):
                month_return += returns_plus_one.iloc[i][strategy] * current_weights[j]
            monthly_portfolio_returns[i] = month_return
        
        # Calculate geometric mean
        geometric_mean = np.prod(monthly_portfolio_returns) ** (1/len(monthly_portfolio_returns))
        
        # Convert to annual return
        annual_geometric_return = geometric_mean ** 12 - 1
        
        print("\n=== GEOMETRIC MEAN CALCULATION ===")
        print(f"Monthly geometric mean: {geometric_mean-1:.6f}")
        print(f"Annual geometric return: {annual_geometric_return:.6f} ({annual_geometric_return*100:.2f}%)")
        
        # Calculate net geometric return
        net_geometric_return = annual_geometric_return - 0.05
        print(f"Net geometric return (after 5% fee): {net_geometric_return:.6f} ({net_geometric_return*100:.2f}%)")
        
        # Try another geometric mean calculation method
        print("\n=== ALTERNATIVE GEOMETRIC MEAN CALCULATION ===")
        # Calculate monthly returns for the portfolio
        portfolio_monthly_returns = np.zeros(len(returns_filled))
        for i in range(len(returns_filled)):
            for j, strategy in enumerate(available_strategies):
                portfolio_monthly_returns[i] += returns_filled.iloc[i][strategy] * current_weights[j]
        
        # Calculate geometric mean of (1+r)
        geo_mean = np.prod(1 + portfolio_monthly_returns) ** (1/len(portfolio_monthly_returns))
        annual_geo_return = geo_mean ** 12 - 1
        
        print(f"Alternative monthly geometric mean: {geo_mean-1:.6f}")
        print(f"Alternative annual geometric return: {annual_geo_return:.6f} ({annual_geo_return*100:.2f}%)")
        
        # Calculate net geometric return
        net_geo_return = annual_geo_return - 0.05
        print(f"Alternative net geometric return (after 5% fee): {net_geo_return:.6f} ({net_geo_return*100:.2f}%)")
        
    except Exception as e:
        print(f"Error calculating geometric mean: {e}")
    
    # Save the results to a text file for easier viewing
    with open("app_calculation_analysis.txt", "w") as f:
        f.write("App Portfolio Return Calculation Extraction\n")
        f.write("=========================================\n\n")
        f.write(f"Using the following weights:\n")
        for s, w in zip(available_strategies, current_weights):
            f.write(f"{s}: {w:.4f} ({w*100:.1f}%)\n")
        
        f.write("\n=== ARITHMETIC MEAN CALCULATION (APP METHOD) ===\n")
        f.write("Strategy | Monthly Mean | Annualized | Weight | Contribution\n")
        f.write("-" * 75 + "\n")
        
        total_contribution = 0
        for i, strategy in enumerate(available_strategies):
            monthly_mean = returns_filled[strategy].mean()
            annualized = monthly_mean * 12
            weight = current_weights[i]
            contribution = annualized * weight
            total_contribution += contribution
            f.write(f"{strategy} | {monthly_mean:.6f} | {annualized:.6f} ({annualized*100:.2f}%) | {weight:.4f} ({weight*100:.1f}%) | {contribution:.6f} ({contribution*100:.2f}%)\n")
        
        f.write("-" * 75 + "\n")
        f.write(f"Total portfolio return: {total_contribution:.6f} ({total_contribution*100:.2f}%)\n")
        f.write(f"Net return (after 5% fee): {net_return:.6f} ({net_return*100:.2f}%)\n")
        
        try:
            f.write("\n=== GEOMETRIC MEAN CALCULATION ===\n")
            f.write(f"Monthly geometric mean: {geometric_mean-1:.6f}\n")
            f.write(f"Annual geometric return: {annual_geometric_return:.6f} ({annual_geometric_return*100:.2f}%)\n")
            f.write(f"Net geometric return (after 5% fee): {net_geometric_return:.6f} ({net_geometric_return*100:.2f}%)\n")
            
            f.write("\n=== ALTERNATIVE GEOMETRIC MEAN CALCULATION ===\n")
            f.write(f"Alternative monthly geometric mean: {geo_mean-1:.6f}\n")
            f.write(f"Alternative annual geometric return: {annual_geo_return:.6f} ({annual_geo_return*100:.2f}%)\n")
            f.write(f"Alternative net geometric return (after 5% fee): {net_geo_return:.6f} ({net_geo_return*100:.2f}%)\n")
        except:
            f.write("\nGeometric mean calculation failed.\n")
    
    print(f"\nResults saved to app_calculation_analysis.txt")

if __name__ == "__main__":
    extract_app_calculation()
