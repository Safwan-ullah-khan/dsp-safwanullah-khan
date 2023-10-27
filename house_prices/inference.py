import pandas as pd
import numpy as np
import joblib
from house_prices.preprocess import scaler_func, encoder_func, join_df

loaded_model = joblib.load('../models/model.joblib')
useful_features = ['Foundation', 'KitchenQual', 'TotRmsAbvGrd', 'WoodDeckSF', 'YrSold', '1stFlrSF']



#### Duplicates Drop function

def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    data = data[~data[useful_features].duplicated(keep='first')]
    data = data.reset_index(drop=True)
    return data

#### Preprocessing function which scales, encodes and rejoins the dataset

def preprocessor_func(data: pd.DataFrame)-> pd.DataFrame:
    continuous_features_df = scaler_func(data)
    categorical_features_df = encoder_func(data)
    final_df = join_df(continuous_features_df, categorical_features_df)
    return final_df


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    df_test = input_data[useful_features]
    df_test = drop_duplicates(df_test)
    df_test = df_test.dropna()
    final_df = preprocessor_func(df_test)
    return loaded_model.predict(final_df)