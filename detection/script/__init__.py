import sys 
import ast
import pandas as pd 
sys.path.append('../')

# Read the CSV file
def df_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    # Select the relevant columns
    df = df[['filename', 'coordinate']]
    # Convert the string representation of the list into an actual list
    df['coordinate'] = df['coordinate'].apply(ast.literal_eval)
    return df