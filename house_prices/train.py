import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from house_prices.preprocess import scaler_func, encoder_func, join_df


label_col = 'SalePrice'
useful_features = ['Foundation', 'KitchenQual',
                   'TotRmsAbvGrd', 'WoodDeckSF',
                   'YrSold', '1stFlrSF']


def scaler_def(data: pd.DataFrame) -> StandardScaler:
    continuous_columns = (data[useful_features]
                          .select_dtypes(include='number')
                          .columns)
    scaler = StandardScaler()
    scaler.fit(data[continuous_columns])
    joblib.dump(scaler, '../models/scaler.joblib')
    return scaler


def encoder_def(data: pd.DataFrame) -> OneHotEncoder:
    categorical_columns = (data[useful_features]
                           .select_dtypes(include='object')
                           .columns)
    encoder = OneHotEncoder()
    encoder.fit(data[categorical_columns])
    joblib.dump(encoder, '../models/encoder.joblib')
    return encoder

# Duplicates Drop function


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    data = data[~data[useful_features].duplicated(keep='first')]
    data = data.reset_index(drop=True)
    return data

# Dataset Split function


def dataset_train_test_split(
        data: pd.DataFrame,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features, target = data.drop(columns=[label_col]), data[label_col]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target,
        test_size=0.33, random_state=42)
    return x_train[useful_features], x_test[useful_features], y_train, y_test

# Preprocessing function which scales, encodes and rejoins the dataset


def preprocessor_func(
        data: pd.DataFrame,
        scaler: StandardScaler,
        encoder: OneHotEncoder
        ) -> pd.DataFrame:
    continuous_features_df = scaler_func(data, scaler)
    categorical_features_df = encoder_func(data, encoder)
    final_df = join_df(continuous_features_df, categorical_features_df)
    return final_df


def model_func(data: pd.DataFrame, y_train: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(data, y_train)
    joblib.dump(model, '../models/model.joblib')
    return model

# model evaluation function


def compute_rmse(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        precision: int = 2
        ) -> float:
    rmse = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmse, precision)


def predict_func(
        data: pd.DataFrame,
        y_test: np.ndarray,
        model: LinearRegression
        ) -> float:
    y_pred = model.predict(data)
    y_pred[y_pred < 0] = 0
    rmse = compute_rmse(y_test, y_pred)
    return rmse

# Function for the model training part


def build_model(data: pd.DataFrame):
    df = drop_duplicates(data)
    X_train, X_test, y_train, y_test = dataset_train_test_split(df)
    scaler, encoder = scaler_def(X_train), encoder_def(X_train)
    final_train_df = preprocessor_func(X_train, scaler, encoder)
    final_test_df = preprocessor_func(X_test, scaler, encoder)
    model = model_func(final_train_df, y_train)
    rmse = str(predict_func(final_test_df, y_test,  model))
    return {'rmse': rmse}
