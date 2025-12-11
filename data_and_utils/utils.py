import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os

def load_data_hourly():
    """
    Reads the training dataset from the specified path.

    Returns:
        pd.DataFrame: dataset.
    """
    data = pd.read_csv("data_and_utils\data\hour.csv", index_col=0)
    return data

def load_data_daily():
    """
    Reads the training dataset from the specified path.

    Returns:
        pd.DataFrame: dataset.
    """
    data = pd.read_csv("data_and_utils\data\day.csv", index_col=0)
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