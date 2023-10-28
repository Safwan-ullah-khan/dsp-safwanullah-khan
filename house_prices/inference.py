import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from house_prices.preprocess import scaler_func, encoder_func, join_df


useful_features = ['Foundation', 'KitchenQual',
                   'TotRmsAbvGrd', 'WoodDeckSF',
                   'YrSold', '1stFlrSF']


def object_loading() -> tuple[StandardScaler, OneHotEncoder, LinearRegression]:
    scaler = joblib.load('../models/scaler.joblib')
    encoder = joblib.load('../models/encoder.joblib')
    model = joblib.load('../models/model.joblib')
    return scaler, encoder, model

# Duplicates Drop function


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    data = data[~data[useful_features].duplicated(keep='first')]
    data = data.reset_index(drop=True)
    return data

# Preprocessing function which scales, encodes and rejoins the dataset


def preprocessor_func(data: pd.DataFrame, scaler: StandardScaler,
                      encoder: OneHotEncoder
                      ) -> pd.DataFrame:
    continuous_features_df = scaler_func(data, scaler)
    categorical_features_df = encoder_func(data, encoder)
    final_df = join_df(continuous_features_df, categorical_features_df)
    return final_df


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    df_test = input_data[useful_features]
    df_test = drop_duplicates(df_test)
    df_test = df_test.dropna()
    scaler, encoder, model = object_loading()
    final_df = preprocessor_func(df_test, scaler, encoder)
    return model.predict(final_df)
