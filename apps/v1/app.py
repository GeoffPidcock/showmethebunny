import os
import pandas as pd
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def load_showbags_csv():
    """
    Load the showbags.csv file using pandas.
    
    Returns:
        pandas.DataFrame: A dataframe containing the showbags data.
    """
    try:
        # Path to the CSV file (use the local one in the app directory)
        csv_path = "showbags.csv"
        
        # Load the CSV into a pandas DataFrame
        showbags_df = pd.read_csv(csv_path)
        
        # Basic validation
        logger.info(f"Successfully loaded {len(showbags_df)} showbags from CSV")
        logger.info(f"CSV columns: {', '.join(showbags_df.columns)}")
        
        return showbags_df
    
    except Exception as e:
        logger.error(f"Error loading showbags CSV: {str(e)}")
        raise

# Test the function if this script is run directly
if __name__ == "__main__":
    print("Testing CSV data loading...")
    showbags_df = load_showbags_csv()
    print(f"Loaded {len(showbags_df)} showbags")
    print(f"First 5 showbags:")
    print(showbags_df.head())
    print("\nCSV columns:")
    for col in showbags_df.columns:
        print(f" - {col}")
