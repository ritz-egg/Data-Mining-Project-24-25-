import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns


# MISSING DATA FUNCTION
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


# IQR OUTLIER FUNCTION
def remove_outliers_iqr(df, columns, threshold=1.5):
    rows_removed = {}  
    total_removed=0
    initial_rows = df.shape[0]
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        removed = df[~((df[column] >= lower_bound) & (df[column] <= upper_bound))].shape[0]
        rows_removed[column] = removed
        total_removed += removed
        
        print(f'Upper Bound for {column}: {upper_bound}')
        print(f'Lower Bound for {column}: {lower_bound}')
        print(50*'-')

        
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f'Rows removed for each:{rows_removed}')
    print(f'Total removed:{total_removed}')
    print(f'Percentage removed:{round((total_removed/initial_rows)*100,4)}%')
    return df


# PERCENTILE OUTLIER FUNCTION
def remove_outliers_percentile(df, columns, lower_percentile=0, upper_percentile=95):
    rows_removed = {}  # Dictionary to store the number of rows removed per column
    total_removed = 0
    initial_rows = df.shape[0]
    
    for column in columns:
        lower_bound = df[column].quantile(lower_percentile / 100)
        upper_bound = df[column].quantile(upper_percentile / 100)
        
        removed = df[~((df[column] >= lower_bound) & (df[column] <= upper_bound))].shape[0]
        rows_removed[column] = removed
        total_removed += removed
        
        print(f'Upper Bound for {column}: {upper_bound}')
        print(f'Lower Bound for {column}: {lower_bound}')
        print(50 * '-')
        
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    print(f'Rows removed for each: {rows_removed}')
    print(f'Total removed: {total_removed}')
    print(f'Percentage removed: {round((total_removed / initial_rows) * 100, 4)}%')
    
    return df


# DBSCAN OUTLIERS FUNCTION
def remove_outliers_dbscan(df, columns, eps=0.5, min_samples=5):
    rows_removed = {}  
    total_removed = 0
    initial_rows = df.shape[0]

    for column in columns:
        column_data = df[[column]]
        
        db = DBSCAN(eps=eps, min_samples=min_samples)
        df['outlier'] = db.fit_predict(column_data)  

        removed = df[df['outlier'] == -1].shape[0]
        rows_removed[column] = removed
        total_removed += removed

        print(f"Outliers in {column}: {removed}")

        df_cleaned = df[df['outlier'] != -1].drop(columns='outlier')

    print(f"\nTotal rows removed: {total_removed}")
    print(f"Percentage of data removed: {round((total_removed / initial_rows) * 100, 4)}%")

    print("\nValue after removing outliers:")
    for column in columns:
        print(f"Maximum value in {column}: {df_cleaned[column].max()}")
        print(f"Minimum value in {column}: {df_cleaned[column].min()}")
        print(50*'-')

    return df_cleaned


# Z-SCORE OUTLIER
def remove_outliers_zscore(df, column, threshold=3):
    rows_removed = {}  
    total_removed = 0
    initial_rows = df.shape[0]


    z_scores = zscore(df[column])

    outliers = np.abs(z_scores) > threshold

    removed = df[outliers].shape[0]
    rows_removed[column] = removed
    total_removed += removed

    print(f"Total outliers in column {column}: {removed}")

    df_cleaned = df[~outliers]

    print(f"Percentage of data removed: {round((total_removed / initial_rows) * 100, 4)}%")
    
    return df_cleaned



# -----------------------------------------------------------------------------------------------
# Functions for plotting


def plot_distribution_and_boxplot(df, columns_with_outliers, w=15,h=12):
    """
    Plots the distribution and boxplot for a list of columns with outliers.
    """
    plt.figure(figsize=(w, h))

    for i, column in enumerate(columns_with_outliers):
        # Distribution plot
        plt.subplot(len(columns_with_outliers), 2, 2 * i + 1)
        sns.histplot(df[column], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

        # Boxplot
        plt.subplot(len(columns_with_outliers), 2, 2 * i + 2)
        sns.boxplot(x=df[column], color='orange')
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)

    plt.tight_layout()
    plt.show()
