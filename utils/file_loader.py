# load the necessary modules
import os
import textract
import pandas as pd
from typing import List, Dict, Any, Union

SUPPORTED_EXT_TYPES = ['.csv', '.xlsx', '.xls', '.pdf', '.txt']

def load_file(file_path:str) -> pd.DataFrame:
    """
    Load a file and return its content as a DataFrame.

    Args:
        file_path (str): The path to the file to be loaded.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    print(f"Loading file: {file_path} with extension: {ext}")
    if ext not in SUPPORTED_EXT_TYPES:
        raise ValueError(f"Unsupported file type: {ext}. Supported types are: {SUPPORTED_EXT_TYPES}")
    
    if ext in ['.csv', '.xlsx', '.xls']:
        return pd.read_csv(file_path) if ext == '.csv' else pd.read_excel(file_path)
    
    elif ext == '.pdf':
        text = textract.process(file_path).decode('utf-8')
        return pd.DataFrame({'content': [text]})
    
    elif ext == '.txt':
        with open(file_path, mode='r', encoding='utf-8') as file:
            text = file.read()
        return pd.DataFrame({'content': [text]})
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported types are: {SUPPORTED_EXT_TYPES}")

    # return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    file_path = r"C:\\Users\\Priya Bhaskar\\OneDrive\\Documents\\ml_agent_project\\data\\Bengaluru_House_Data.csv"
    df = load_file(file_path)
    print(df.head())
