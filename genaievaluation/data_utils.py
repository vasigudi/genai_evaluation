# genaievaluation/genaievaluation/data_utils.py

import pandas as pd

def preprocess_data(df):
    """
    Preprocess the DataFrame by stripping whitespace from column names
    and removing any rows with missing values in key columns.
    
    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df.columns = df.columns.str.strip()
    required_columns = ['id', 'prompt', 'ground_truth', 'generated_response']
    df = df.dropna(subset=required_columns)
    return df