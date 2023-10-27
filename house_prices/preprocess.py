import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_log_error

loaded_scaler = joblib.load('../models/scaler.joblib')
loaded_encoder = joblib.load('../models/encoder.joblib')

useful_features = ['Foundation', 'KitchenQual', 'TotRmsAbvGrd', 'WoodDeckSF', 'YrSold', '1stFlrSF']





#### Standard scaling function

def scaler_func(data: pd.DataFrame)-> pd.DataFrame:
    continuous_columns = data[useful_features].select_dtypes(include='number').columns
    scaled_columns = loaded_scaler.transform(data[continuous_columns])
    continuous_features_df = pd.DataFrame(data=scaled_columns, columns=continuous_columns)
    return continuous_features_df

#### One hot encoding function

def encoder_func(data: pd.DataFrame)-> pd.DataFrame:
    categorical_columns = data[useful_features].select_dtypes(include='object').columns
    categorical_columns_list = categorical_columns.tolist()
    labels = loaded_encoder.get_feature_names_out(categorical_columns_list)
    encoded_data = loaded_encoder.transform(data[categorical_columns_list]).toarray()
    categorical_features_df = pd.DataFrame(data=encoded_data, columns=labels)
    return categorical_features_df

#### Function to join continuous scaled and categorical scaled datasets

def join_df(df1: pd.DataFrame, df2: pd.DataFrame)->pd.DataFrame:
    final_df = df1.join(df2)
    return final_df

