import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob

def extract_exact_return():
    """
    Extract the exact calculation for the 11.97% return shown in the UI.
    This script attempts to replicate the exact calculation used in the app.
    """
    print("Extracting Exact Calculation for 11.97% Return")
    print("=============================================\n")
    
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
    print(f"Date range: {monthly_data['Month'].min()} to {monthly_data['Month'].max()}")
    
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
        
        # Extract strategy weights
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
    
    # Try different time periods
    periods = {
        "ITD": None,  # All data
        "T12M": 12,   # Last 12 months
        "T6M": 6,     # Last 6 months
        "T24M": 24    # Last 24 months
    }
    
    # Set SHORT TERM yield
    short_term_yield = 4.2  # 4.2% annual yield
    
    # Create weights array in the same order as the strategies
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
    
    # Try each period
    print("\n=== TESTING DIFFERENT TIME PERIODS ===")
    for period_name, months in periods.items():
        print(f"\n--- {period_name} PERIOD ---")
        
        # Filter data for the period
        if months is None:
            # Use all data for ITD
            filtered_data = monthly_data.copy()
        else:
            # Get the most recent date
            most_recent_date = monthly_data['Month'].max()
            # Calculate cutoff date
            cutoff_date = most_recent_date - pd.DateOffset(months=months)
            # Filter data
            filtered_data = monthly_data[monthly_data['Month'] >= cutoff_date]
        
        print(f"Using {len(filtered_data)} months of data from {filtered_data['Month'].min()} to {filtered_data['Month'].max()}")
        
        # Set the Month as index
        filtered_data.set_index('Month', inplace=True)
        
        # Extract relevant columns
        monthly_returns = filtered_data[available_strategies].copy()
        
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
        
        # Set SHORT TERM to use a fixed yield
        if 'SHORT TERM F1' in monthly_returns.columns:
            monthly_returns['SHORT TERM F1'] = short_term_yield / 100.0 / 12
            print(f"Set SHORT TERM F1 to fixed monthly yield: {short_term_yield/100.0/12:.6f} ({short_term_yield}% annual)")
        
        # Handle missing values
        returns_filled = monthly_returns.copy()
        for col in returns_filled.columns:
            returns_filled[col] = returns_filled[col].fillna(returns_filled[col].mean())
        
        # Calculate arithmetic mean return
        print("\n=== ARITHMETIC MEAN CALCULATION ===")
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
        
        # Check if this matches the target 11.97%
        if abs(net_return * 100 - 11.97) < 0.1:
            print(f"\n*** MATCH FOUND! This calculation produces {net_return*100:.2f}%, which matches the target 11.97% ***")
        
        # Try geometric mean calculation
        try:
            print("\n=== GEOMETRIC MEAN CALCULATION ===")
            
            # Calculate portfolio returns for each month
            portfolio_monthly_returns = np.zeros(len(returns_filled))
            for i in range(len(returns_filled)):
                for j, strategy in enumerate(available_strategies):
                    portfolio_monthly_returns[i] += returns_filled.iloc[i][strategy] * current_weights[j]
            
            # Calculate geometric mean
            geo_mean = np.prod(1 + portfolio_monthly_returns) ** (1/len(portfolio_monthly_returns))
            annual_geo_return = geo_mean ** 12 - 1
            
            print(f"Monthly geometric mean: {geo_mean-1:.6f}")
            print(f"Annual geometric return: {annual_geo_return:.6f} ({annual_geo_return*100:.2f}%)")
            
            # Calculate net geometric return
            net_geo_return = annual_geo_return - 0.05
            print(f"Net geometric return (after 5% fee): {net_geo_return:.6f} ({net_geo_return*100:.2f}%)")
            
            # Check if this matches the target 11.97%
            if abs(net_geo_return * 100 - 11.97) < 0.1:
                print(f"\n*** MATCH FOUND! Geometric calculation produces {net_geo_return*100:.2f}%, which matches the target 11.97% ***")
        except Exception as e:
            print(f"Error calculating geometric mean: {e}")
    
    # Try a different approach - weighted geometric mean
    print("\n=== TRYING WEIGHTED GEOMETRIC MEAN APPROACH ===")
    
    # Use all data for this approach
    monthly_data_copy = monthly_data.copy()
    monthly_data_copy.set_index('Month', inplace=True)
    monthly_returns_all = monthly_data_copy[available_strategies].copy()
    
    # Convert to numeric values
    for col in available_strategies:
        if monthly_returns_all[col].dtype == 'object':
            try:
                if monthly_returns_all[col].astype(str).str.contains('%').any():
                    monthly_returns_all[col] = monthly_returns_all[col].str.rstrip('%').astype(float) / 100.0
                else:
                    monthly_returns_all[col] = pd.to_numeric(monthly_returns_all[col], errors='coerce')
            except:
                monthly_returns_all[col] = pd.to_numeric(monthly_returns_all[col], errors='coerce')
    
    # Set SHORT TERM to use a fixed yield
    if 'SHORT TERM F1' in monthly_returns_all.columns:
        monthly_returns_all['SHORT TERM F1'] = short_term_yield / 100.0 / 12
    
    # Handle missing values
    returns_filled_all = monthly_returns_all.copy()
    for col in returns_filled_all.columns:
        returns_filled_all[col] = returns_filled_all[col].fillna(returns_filled_all[col].mean())
    
    # Calculate geometric mean for each strategy
    print("Strategy | Geometric Mean | Annualized | Weight | Contribution")
    print("-" * 75)
    
    total_weighted_geo = 0
    for i, strategy in enumerate(available_strategies):
        # Calculate geometric mean for this strategy
        strategy_returns = returns_filled_all[strategy]
        geo_mean = np.prod(1 + strategy_returns) ** (1/len(strategy_returns))
        annual_geo = geo_mean ** 12 - 1
        
        # Apply weight
        weight = current_weights[i]
        contribution = annual_geo * weight
        total_weighted_geo += contribution
        
        print(f"{strategy} | {geo_mean-1:.6f} | {annual_geo:.6f} ({annual_geo*100:.2f}%) | {weight:.4f} ({weight*100:.1f}%) | {contribution:.6f} ({contribution*100:.2f}%)")
    
    print("-" * 75)
    print(f"Total weighted geometric return: {total_weighted_geo:.6f} ({total_weighted_geo*100:.2f}%)")
    
    # Calculate net return
    net_weighted_geo = total_weighted_geo - 0.05
    print(f"Net weighted geometric return: {net_weighted_geo:.6f} ({net_weighted_geo*100:.2f}%)")
    
    # Check if this matches the target 11.97%
    if abs(net_weighted_geo * 100 - 11.97) < 0.1:
        print(f"\n*** MATCH FOUND! Weighted geometric calculation produces {net_weighted_geo*100:.2f}%, which matches the target 11.97% ***")
    
    # Save results to a file
    with open("exact_return_analysis.txt", "w") as f:
        f.write("Extracting Exact Calculation for 11.97% Return\n")
        f.write("=============================================\n\n")
        f.write("Portfolio weights:\n")
        for s, w in zip(available_strategies, current_weights):
            f.write(f"{s}: {w:.4f} ({w*100:.1f}%)\n")
        
        f.write("\nResults by time period:\n")
        for period_name, months in periods.items():
            f.write(f"\n--- {period_name} PERIOD ---\n")
            
            # Filter data for the period
            if months is None:
                # Use all data for ITD
                filtered_data = monthly_data.copy()
            else:
                # Get the most recent date
                most_recent_date = monthly_data['Month'].max()
                # Calculate cutoff date
                cutoff_date = most_recent_date - pd.DateOffset(months=months)
                # Filter data
                filtered_data = monthly_data[monthly_data['Month'] >= cutoff_date]
            
            f.write(f"Using {len(filtered_data)} months of data\n")
            
            # Set the Month as index
            filtered_data.set_index('Month', inplace=True)
            
            # Extract relevant columns
            monthly_returns = filtered_data[available_strategies].copy()
            
            # Convert to numeric values and handle missing values
            returns_filled = monthly_returns.copy()
            for col in returns_filled.columns:
                if returns_filled[col].dtype == 'object':
                    try:
                        if returns_filled[col].astype(str).str.contains('%').any():
                            returns_filled[col] = returns_filled[col].str.rstrip('%').astype(float) / 100.0
                        else:
                            returns_filled[col] = pd.to_numeric(returns_filled[col], errors='coerce')
                    except:
                        returns_filled[col] = pd.to_numeric(returns_filled[col], errors='coerce')
                returns_filled[col] = returns_filled[col].fillna(returns_filled[col].mean())
            
            # Set SHORT TERM
            if 'SHORT TERM F1' in returns_filled.columns:
                returns_filled['SHORT TERM F1'] = short_term_yield / 100.0 / 12
            
            # Calculate arithmetic mean
            total_contribution = 0
            for i, strategy in enumerate(available_strategies):
                monthly_mean = returns_filled[strategy].mean()
                annualized = monthly_mean * 12
                weight = current_weights[i]
                contribution = annualized * weight
                total_contribution += contribution
            
            net_return = total_contribution - 0.05
            f.write(f"Arithmetic mean net return: {net_return*100:.2f}%\n")
            
            # Calculate geometric mean
            try:
                portfolio_monthly_returns = np.zeros(len(returns_filled))
                for i in range(len(returns_filled)):
                    for j, strategy in enumerate(available_strategies):
                        portfolio_monthly_returns[i] += returns_filled.iloc[i][strategy] * current_weights[j]
                
                geo_mean = np.prod(1 + portfolio_monthly_returns) ** (1/len(portfolio_monthly_returns))
                annual_geo_return = geo_mean ** 12 - 1
                net_geo_return = annual_geo_return - 0.05
                f.write(f"Geometric mean net return: {net_geo_return*100:.2f}%\n")
            except:
                f.write("Geometric mean calculation failed\n")
    
    print(f"\nResults saved to exact_return_analysis.txt")

if __name__ == "__main__":
    extract_exact_return()
