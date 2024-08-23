# genaievaluation/genaievaluation/evaluate.py
import pandas as pd
from .data_utils import preprocess_data
from .metrics_utils import compute_data

def run_evaluation(df):
    """
    Run the evaluation on the provided DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to evaluate.
    
    Returns:
        pd.DataFrame: DataFrame with evaluation metrics added.
    """
    # Preprocess the data
    df = preprocess_data(df)
    
    # Compute metrics
    result_df = compute_data(df)
    
    return result_df