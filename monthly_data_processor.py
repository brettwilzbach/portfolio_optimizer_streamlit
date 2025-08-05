import pandas as pd
import numpy as np
import os

def calculate_annual_returns_from_monthly(monthly_data, period="ITD RoA"):
    """
    Calculate annual returns from monthly data for each strategy.
    
    Parameters:
    -----------
    monthly_data : pd.DataFrame
        DataFrame with monthly returns for each strategy, with 'Month' as a column
    period : str
        Period to calculate returns for ('ITD RoA', 'T12M RoA', or 'T6M RoA')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with annual returns for each strategy
    """
    if monthly_data is None or monthly_data.empty:
        print("No monthly data available for annual return calculation")
        return pd.DataFrame()
    
    # Make a copy to avoid modifying the original
    data = monthly_data.copy()
    
    # Ensure Month column exists and is a datetime
    if 'Month' not in data.columns:
        print("Monthly data must have a 'Month' column")
        return pd.DataFrame()
    
    # Convert Month to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(data['Month']):
        try:
            data['Month'] = pd.to_datetime(data['Month'])
        except Exception as e:
            print(f"Error converting Month to datetime: {e}")
            return pd.DataFrame()
    
    # Filter data based on the selected period
    if period == "T12M RoA":
        # Get the most recent date in the data
        most_recent_date = data['Month'].max()
        # Filter to only include the last 12 months
        twelve_months_ago = most_recent_date - pd.DateOffset(months=12)
        data = data[data['Month'] >= twelve_months_ago]
        print(f"Filtered monthly data to last 12 months: {len(data)} rows remaining")
    elif period == "T6M RoA":
        # Get the most recent date in the data
        most_recent_date = data['Month'].max()
        # Filter to only include the last 6 months
        six_months_ago = most_recent_date - pd.DateOffset(months=6)
        data = data[data['Month'] >= six_months_ago]
        print(f"Filtered monthly data to last 6 months: {len(data)} rows remaining")
    else:
        # For ITD RoA, use all available data
        print(f"Using all available monthly data for ITD: {len(data)} rows")
    
    # Set Month as index
    data.set_index('Month', inplace=True)
    
    # Get list of strategy columns (exclude non-strategy columns)
    strategy_columns = [col for col in data.columns if col not in ['Total', 'AGGREGATE']]
    
    # Calculate annual returns for each strategy
    annual_returns = {}
    
    for strategy in strategy_columns:
        # Calculate arithmetic mean of monthly returns
        monthly_mean = data[strategy].mean()
        # Annualize by multiplying by 12
        annual_return = monthly_mean * 12
        annual_returns[strategy] = annual_return
    
    # Create DataFrame with annual returns
    annual_returns_df = pd.DataFrame({
        'Strategy': list(annual_returns.keys()),
        'RoA': list(annual_returns.values())
    })
    
    return annual_returns_df

def map_monthly_columns_to_strategies(monthly_data, strategies):
    """
    Map monthly data columns to strategy names.
    
    Parameters:
    -----------
    monthly_data : pd.DataFrame
        DataFrame with monthly returns
    strategies : list
        List of strategy names to map to
        
    Returns:
    --------
    dict
        Dictionary mapping strategy names to monthly data columns
    """
    if monthly_data is None or monthly_data.empty:
        return {}
    
    # Strategy mapping for monthly data columns
    strategy_mapping = {
        'AIRCRAFT F1': 'AIRCRAFT F1',
        'CMBS F1': 'CMBS F1',
        'SHORT TERM': 'SHORT TERM F1',  # Note the F1 suffix in the data
        'CLO F1': 'CLO F1',
        'ABS F1': 'ABS F1'
    }
    
    # Create a mapping from our strategy names to the column names in the data
    column_mapping = {strategy_mapping.get(s, s): s for s in strategies if strategy_mapping.get(s, s) in monthly_data.columns}
    
    return column_mapping

def prepare_monthly_data_for_calculations(monthly_data, strategies, short_term_yield=None):
    """
    Prepare monthly data for calculations by cleaning, filtering, and mapping to strategies.
    
    Parameters:
    -----------
    monthly_data : pd.DataFrame
        DataFrame with monthly returns
    strategies : list
        List of strategy names to include
    short_term_yield : float, optional
        Annual yield for SHORT TERM strategy (in percentage)
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and prepared monthly returns data
    """
    if monthly_data is None or monthly_data.empty:
        return None
    
    # Make a copy to avoid modifying the original
    data = monthly_data.copy()
    
    # Strategy mapping for monthly data columns
    strategy_mapping = {
        'AIRCRAFT F1': 'AIRCRAFT F1',
        'CMBS F1': 'CMBS F1',
        'SHORT TERM': 'SHORT TERM F1',  # Note the F1 suffix in the data
        'CLO F1': 'CLO F1',
        'ABS F1': 'ABS F1'
    }
    
    # Create a list of columns needed from the monthly data
    needed_columns = [strategy_mapping.get(s, s) for s in strategies if strategy_mapping.get(s, s) in data.columns]
    
    if len(needed_columns) > 0:
        # Convert percentage strings to float values and handle blank cells
        for col in needed_columns:
            if data[col].dtype == 'object':
                # First convert empty strings to NaN
                data[col] = data[col].replace('', np.nan)
                # Then convert percentage strings to float values for non-NaN cells
                data[col] = pd.to_numeric(
                    data[col].str.rstrip('%') if isinstance(data[col], pd.Series) else data[col], 
                    errors='coerce'
                ) / 100.0
        
        # Set the Month as index
        if 'Month' in data.columns:
            data.set_index('Month', inplace=True)
        
        # Create a mapping from our strategy names to the column names in the data
        column_mapping = {strategy_mapping.get(s, s): s for s in strategies if strategy_mapping.get(s, s) in data.columns}
        
        # Extract and rename columns to match our strategy names
        monthly_returns = data[needed_columns].rename(columns=column_mapping)
        
        # Ensure SHORT TERM is using the exact yield from Portfolio Settings
        if 'SHORT TERM' in monthly_returns.columns and short_term_yield is not None:
            # Set SHORT TERM to use the user-defined annual yield (divided by 12 for monthly)
            monthly_returns['SHORT TERM'] = short_term_yield / 100.0 / 12  # Convert from percentage to decimal and then to monthly
        
        return monthly_returns
    
    return None

def load_monthly_roa_data(monthly_data_file=None):
    """Load the monthly RoA data with error handling"""
    try:
        # If a file was uploaded through the UI, use that first
        if monthly_data_file is not None:
            try:
                monthly_data = pd.read_excel(monthly_data_file)
                print(f"Loaded Monthly RoA data from uploaded file")
                # Convert the Month column to datetime
                if 'Month' in monthly_data.columns:
                    monthly_data['Month'] = pd.to_datetime(monthly_data['Month'])
                return monthly_data
            except Exception as upload_err:
                print(f"Error loading uploaded Monthly RoA data: {upload_err}")
                # Fall back to looking for the file on disk
        
        # Look for the file in several possible locations
        possible_paths = [
            "Aggregate Monthly RoA.xlsx",
            os.path.join(os.getcwd(), "Aggregate Monthly RoA.xlsx"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "Aggregate Monthly RoA.xlsx"),
            "./portfolio_optimizer_streamlit/Aggregate Monthly RoA.xlsx",
            "./Aggregate Monthly RoA.xlsx",
            "C:/Users/bwilzbach/CascadeProjects/Aggregate Monthly RoA.xlsx",
            "C:/Users/bwilzbach/Desktop/Cash Drag Project/Aggregate Monthly RoA.xlsx"
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                print(f"Found Monthly RoA data at: {file_path}")
                break
        
        if file_path is None:
            raise FileNotFoundError(f"Monthly RoA data not found in any of the expected locations: {possible_paths}")
        
        # Load the data
        monthly_data = pd.read_excel(file_path)
        
        # Convert the Month column to datetime
        if 'Month' in monthly_data.columns:
            monthly_data['Month'] = pd.to_datetime(monthly_data['Month'])
        
        return monthly_data
    except Exception as e:
        print(f"Error loading Monthly RoA data: {e}")
        return None


def generate_annual_returns_from_monthly(force_reload=False, period="ITD RoA", monthly_data_file=None):
    """Generate annual returns from monthly data to mimic the RoA Master Sheet structure"""
    import streamlit as st
    
    # Check if we have cached data and force_reload is False
    if not force_reload and 'monthly_roa_data' in st.session_state:
        print("Using cached monthly RoA data")
        monthly_data = st.session_state.monthly_roa_data
    else:
        # Load the monthly data
        monthly_data = load_monthly_roa_data(monthly_data_file)
        if monthly_data is not None:
            # Cache the data for future use
            st.session_state.monthly_roa_data = monthly_data
    
    if monthly_data is None or monthly_data.empty:
        print("No monthly data available")
        return pd.DataFrame()
    
    # Calculate annual returns from monthly data
    annual_returns = calculate_annual_returns_from_monthly(monthly_data, period)
    
    # Process the annual returns to match the RoA Master Sheet structure
    # We need to create a DataFrame with columns: Strategy, Substrategy, ITD RoA, T12M RoA, T6M RoA
    
    # First, create a DataFrame with all the strategies
    strategies = annual_returns['Strategy'].tolist()
    
    # Create a DataFrame with the structure of the RoA Master Sheet
    roa_master = pd.DataFrame({
        'Strategy': strategies,
        'Substrategy': [''] * len(strategies)  # Empty substrategy for main strategies
    })
    
    # Add the annual returns for the selected period
    roa_master[period] = annual_returns['RoA'].values
    
    # Add placeholder columns for other periods if they don't exist
    for p in ['ITD RoA', 'T12M RoA', 'T6M RoA']:
        if p not in roa_master.columns:
            roa_master[p] = roa_master[period]  # Use the selected period as a placeholder
    
    print(f"Generated annual returns from monthly data for {len(strategies)} strategies")
    print(f"Sample of generated annual returns:\n{roa_master.head()}")
    
    return roa_master
