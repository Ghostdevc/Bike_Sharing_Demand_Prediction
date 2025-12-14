import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os
from sklearn.neighbors import LocalOutlierFactor

def load_data_hourly():
    """
    Reads the training dataset from the specified path.

    Returns:
        pd.DataFrame: dataset.
    """
    data = pd.read_csv("data_and_utils\data\hour.csv", index_col=0, parse_dates=['dteday'])
    return data

def load_data_daily():
    """
    Reads the training dataset from the specified path.

    Returns:
        pd.DataFrame: dataset.
    """
    data = pd.read_csv("data_and_utils\data\day.csv", index_col=0, parse_dates=['dteday'])
    return data

def check_df(dataframe):
    """
    Checks the overall structure and key metrics of a DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame to inspect.

    Returns:
        None: Prints shape, data types, head, tail, missing values, and quantiles.
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(5))
    print("##################### Tail #####################")
    print(dataframe.tail(5))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print('##################### Unique Values #####################')
    print(dataframe.nunique())
    print("##################### Duplicates #####################")
    print(dataframe.duplicated().sum())
    print("##################### Quantiles #####################")
    # Uncomment below to include quantile information
    #print(dataframe[[col for col in dataframe.columns if dataframe[col].dtypes != "O"]].quantile([0, 0.05, 0.50, 0.75, 0.95, 0.99, 1]).T)
    print(dataframe.describe().T)


def grab_col_names(dataframe, cat_th=13, car_th=20):
    """
    Identifies categorical, numerical, and cardinal columns in a DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze.
        cat_th (int): Threshold for numerical columns to be considered categorical.
        car_th (int): Threshold for categorical columns to be considered cardinal.

    Returns:
        tuple: A tuple containing:
            - cat_cols (list): Categorical columns.
            - num_cols (list): Numerical columns.
            - cat_but_car (list): Cardinal columns.
    """
    # Categorical columns and categorical-like numerical columns
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Numerical columns excluding categorical-like ones
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # Output summary
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Kategorik bir özellik ile hedef değişken arasındaki ortalama ilişkiyi gösterir.

    Parametreler:
    dataframe : pandas.DataFrame
        Veri seti.
    target : str
        Hedef değişkenin adı.
    categorical_col : str
        Kategorik özelliğin adı.
    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def target_summary_with_num(dataframe, target, numerical_col):
    """
    Sayısal bir özellik ile hedef değişken arasındaki ortalama ilişkiyi gösterir.

    Parametreler:
    dataframe : pandas.DataFrame
        Veri seti.
    target : str
        Hedef değişkenin adı.
    numerical_col : str
        Sayısal özelliğin adı.
    """
    print(dataframe.groupby(target).agg({numerical_col: ['mean', 'median', 'std', 'min', 'max']}), end="\n\n\n")


def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    """
    Identifies highly correlated columns.

    Args:
        dataframe (pd.DataFrame): DataFrame to analyze.
        plot (bool): Whether to display a correlation heatmap.
        corr_th (float): Correlation threshold for identifying columns.

    Returns:
        list: List of columns with high correlation.
    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="vlag")
        plt.show()
    return drop_list


def correlation_matrix(df, cols):
    """
    Visualizes the correlation matrix for specified columns.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
        cols (list): List of columns to include in the correlation matrix.

    Returns:
        None: Displays a heatmap.
    """
    fig = plt.gcf()
    fig.set_size_inches(20, 16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5,
                      annot_kws={'size': 12}, linecolor='w', cmap='vlag')
    plt.show(block=True)


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Calculates the lower and upper thresholds for outlier detection based on the interquartile range (IQR).

    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze.
        col_name (str): Column name for which to calculate thresholds.
        q1 (float): Lower quantile (default is 0.25).
        q3 (float): Upper quantile (default is 0.75).

    Returns:
        tuple: A tuple containing the lower and upper thresholds.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """
    Checks if a column contains outliers based on calculated thresholds.

    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze.
        col_name (str): Column name to check.

    Returns:
        bool: True if outliers exist, False otherwise.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)


def grab_outliers(dataframe, col_name, index=False):
    """
    Retrieves and optionally returns the indices of outliers in a specified column.

    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze.
        col_name (str): Column name to retrieve outliers from.
        index (bool): If True, returns the indices of the outliers.

    Returns:
        pd.Index (optional): Indices of outliers if index=True.
    """
    low, up = outlier_thresholds(dataframe, col_name)


    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])


    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_outlier(dataframe, col_name):
    """
    Removes outliers from a specified column.

    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze.
        col_name (str): Column name to remove outliers from.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def replace_with_thresholds(dataframe, variable):
    """
    Replaces outliers in a specified column with the lower and upper thresholds.

    Args:
        dataframe (pd.DataFrame): The DataFrame to modify.
        variable (str): Column name to replace outliers in.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



