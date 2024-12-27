import pandas as pd


def missing_data(df):
    """
    Gives the count and percentage of missing values for each column in a DataFrame
    """    
    # Number of missing values in each column
    missing_count = df.isnull().sum()
    
    # Percentage of missing values for each column
    missing_percentage = ((missing_count / df.shape[0]) * 100).round(2)
    
    missing_data = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing %': missing_percentage
    })
    
    # Show only columns with missing values
    missing_data = missing_data[missing_data['Missing Count'] > 0]
    
    # Sort in descending order
    missing_data = missing_data.sort_values(by='Missing Count', ascending=False)
    return missing_data