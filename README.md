# Cash Drag Project

This repository contains code and data for the Cash Drag Project, which includes a Portfolio Optimizer tool built with Streamlit.

## Project Structure

- `code/` - Contains Python code files
  - `Portfolio Optimizer 2.0.py` - Main Streamlit application for portfolio optimization
  - `efficient_frontier.py` - Helper functions for portfolio optimization
  
- `data/` - Contains data files
  - `Monthly RoA Total - Correct.xlsx` - Monthly Return on Assets data
  - `Monthly RoA.xlsx` - Monthly Return on Assets data
  - Portfolio holdings files
  
- `run_portfolio_optimizer.bat` - Batch file to run the Portfolio Optimizer application

## Setup on a New Computer

1. Clone this repository to your new computer
2. Install Python 3.8+ if not already installed
3. Install required packages:
   ```
   pip install streamlit pandas numpy matplotlib plotly scipy
   ```
4. Place your data files in the `data/` directory
5. Run the application using the batch file or directly with:
   ```
   streamlit run code/Portfolio\ Optimizer\ 2.0.py
   ```

## Data File Locations

The application looks for data files in several locations. To ensure it works on your new computer:

1. Place all Excel files in the `data/` directory
2. The application will look for files in the following locations:
   - Current working directory
   - The `data/` directory
   - Relative paths from the code directory

## Updating Data

When you have new data files:

1. Place them in the `data/` directory
2. The application will automatically use the most recent files based on naming conventions
3. For monthly data updates, follow the same naming format as existing files
