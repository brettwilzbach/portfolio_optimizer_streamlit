import pandas as pd
import numpy as np
import os
from pathlib import Path

def extract_monthly_returns():
    """
    Extract and analyze the monthly returns data to understand the exact calculations for the 11.97% return.
    """
    print("Monthly Returns Analysis Tool")
    print("============================\n")
    
    # Load the monthly RoA data
    try:
        # Use the Aggregate Monthly RoA.xlsx file directly
        monthly_file = Path(__file__).parent / "Aggregate Monthly RoA.xlsx"
        
        if not monthly_file.exists():
            print(f"Error: Monthly data file not found: {monthly_file}")
            return
            
        print(f"Loading monthly data from: {monthly_file}")
        
        # Load the data
        monthly_data = pd.read_excel(monthly_file)
        print(f"Loaded data with shape: {monthly_data.shape}")
        
        # Print column names
        print(f"Columns: {monthly_data.columns.tolist()}")
        
        # Check for Month column
        if 'Month' not in monthly_data.columns:
            print("Error: 'Month' column not found in the data.")
            # Try to identify a date column
            date_cols = [col for col in monthly_data.columns if 'date' in col.lower() or 'month' in col.lower()]
            if date_cols:
                print(f"Possible date columns: {date_cols}")
            return
        
        # Map strategy names to match the format in the monthly data
        strategy_mapping = {
            'AIRCRAFT F1': 'AIRCRAFT F1',
            'CMBS F1': 'CMBS F1',
            'SHORT TERM': 'SHORT TERM F1',  # Note the F1 suffix in the data
            'CLO F1': 'CLO F1',
            'ABS F1': 'ABS F1'
        }
        
        # Check which strategies exist in the data
        available_strategies = []
        for strategy in strategy_mapping.values():
            if strategy in monthly_data.columns:
                available_strategies.append(strategy)
        
        if not available_strategies:
            print("No matching strategy columns found in the data.")
            print("Available columns:", monthly_data.columns.tolist())
            return
            
        print(f"\nFound {len(available_strategies)} strategies in the data: {available_strategies}")
        
        # Convert Month column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(monthly_data['Month']):
            try:
                monthly_data['Month'] = pd.to_datetime(monthly_data['Month'])
                print("Converted Month column to datetime format.")
            except Exception as e:
                print(f"Warning: Could not convert Month column to datetime: {e}")
                print("Month column sample values:", monthly_data['Month'].head())
        
        # Set the Month as index
        monthly_data.set_index('Month', inplace=True)
        
        # Extract the relevant columns
        monthly_returns = monthly_data[available_strategies]
        
        # Convert percentage strings to float values and handle blank cells
        for col in available_strategies:
            if monthly_returns[col].dtype == 'object':
                print(f"Converting column {col} from {monthly_returns[col].dtype} to numeric...")
                # First convert empty strings to NaN
                monthly_returns[col] = monthly_returns[col].replace('', np.nan)
                # Then convert percentage strings to float values for non-NaN cells
                try:
                    if isinstance(monthly_returns[col], pd.Series) and monthly_returns[col].str.contains('%').any():
                        monthly_returns[col] = monthly_returns[col].str.rstrip('%').astype(float) / 100.0
                    else:
                        monthly_returns[col] = pd.to_numeric(monthly_returns[col], errors='coerce') / 100.0
                except Exception as e:
                    print(f"Error converting {col}: {e}")
                    print(f"Sample values: {monthly_returns[col].head()}")
        
        # Set SHORT TERM to use a fixed yield (4.2% annual, 0.35% monthly)
        short_term_yield = 4.2
        if 'SHORT TERM F1' in monthly_returns.columns:
            monthly_returns['SHORT TERM F1'] = short_term_yield / 100.0 / 12
            print(f"Set SHORT TERM F1 to fixed monthly yield: {short_term_yield/100.0/12:.6f} ({short_term_yield}% annual)")
        
        # Define current weights (these should match what's in the app)
        # For demonstration, using placeholder weights
        weights = {
            'AIRCRAFT F1': 0.25,
            'CMBS F1': 0.25,
            'SHORT TERM F1': 0.20,
            'CLO F1': 0.15,
            'ABS F1': 0.15
        }
        
        # Create weights array in the same order as the strategies
        current_weights = np.array([weights.get(s, 0.0) for s in available_strategies])
        
        # Normalize weights to sum to 1
        current_weights = current_weights / current_weights.sum()
        
        print("\nUsing the following weights:")
        for s, w in zip(available_strategies, current_weights):
            print(f"{s}: {w:.4f} ({w*100:.1f}%)")
        
        # Print detailed information about the monthly returns
        print("\nMonthly Returns Data Summary:")
        print(f"Date range: {monthly_returns.index.min()} to {monthly_returns.index.max()}")
        print(f"Number of months: {len(monthly_returns)}")
        
        # Handle missing values in the returns data
        returns_filled = monthly_returns.copy()
        
        # Check if any columns have too many missing values
        missing_pct = returns_filled.isna().mean()
        print("\nMissing data percentage by strategy:")
        for col in returns_filled.columns:
            print(f"{col}: {missing_pct[col]*100:.1f}% missing")
        
        # Replace NaN with column mean for better accuracy
        for col in returns_filled.columns:
            if missing_pct[col] > 0.5:
                print(f"WARNING: {col} has {missing_pct[col]*100:.1f}% missing data - using available data but results may be less reliable")
            returns_filled[col] = returns_filled[col].fillna(returns_filled[col].mean())
        
        # Calculate expected returns (annualized)
        expected_returns = returns_filled.mean() * 12
        portfolio_return = np.sum(expected_returns * current_weights)
        
        # Calculate portfolio volatility (annualized)
        cov_matrix = returns_filled.cov() * 12
        portfolio_volatility = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))
        
        # Print the contribution of each strategy to the portfolio return
        print("\nDETAILED DEBUG - Exact Calculations for Portfolio Return:")
        print("Strategy | Monthly Mean | Annualized | Weight | Contribution")
        print("-" * 75)
        total_contribution = 0
        for i, strategy in enumerate(available_strategies):
            monthly_mean = returns_filled[strategy].mean()
            annualized = monthly_mean * 12
            weight = current_weights[i]
            contribution = annualized * weight
            total_contribution += contribution
            print(f"{strategy} | {monthly_mean:.6f} | {annualized:.6f} ({annualized*100:.2f}%) | {weight:.6f} ({weight*100:.1f}%) | {contribution:.6f} ({contribution*100:.2f}%)")
        
        print("-" * 75)
        print(f"Total portfolio return (sum of contributions): {total_contribution:.6f} ({total_contribution*100:.2f}%)")
        print(f"Portfolio return calculation: {portfolio_return:.6f} ({portfolio_return*100:.2f}%)")
        print(f"Portfolio volatility: {portfolio_volatility:.6f} ({portfolio_volatility*100:.2f}%)")
        
        # Calculate net return (after 5% fee)
        portfolio_net_return = portfolio_return - 0.05
        print(f"Net return (after 5% fee): {portfolio_net_return:.6f} ({portfolio_net_return*100:.2f}%)")
        
    except Exception as e:
        print(f"Error analyzing monthly returns: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_monthly_returns()
