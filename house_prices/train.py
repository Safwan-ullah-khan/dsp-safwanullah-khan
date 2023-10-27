import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from house_prices.preprocess import scaler_func, encoder_func, join_df


loaded_model = joblib.load('../models/model.joblib')
label_col = 'SalePrice'
useful_features = ['Foundation', 'KitchenQual', 'TotRmsAbvGrd', 'WoodDeckSF', 'YrSold', '1stFlrSF']


#### Duplicates Drop function
def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    data = data[~data[useful_features].duplicated(keep='first')]
    data = data.reset_index(drop=True)
    return data

#### Dataset Split function
def dataset_train_test_split(
    data: pd.DataFrame,
    test_size: float = 0.33,
    random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features, target = data.drop(columns=[label_col]), data[label_col]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)
    return x_train[useful_features], x_test[useful_features],y_train, y_test

#### Preprocessing function which scales, encodes and rejoins the dataset
def preprocessor_func(data: pd.DataFrame)-> pd.DataFrame:
    continuous_features_df = scaler_func(data)
    categorical_features_df = encoder_func(data)
    final_df = join_df(continuous_features_df, categorical_features_df)
    return final_df

#### model evaluation function
def compute_rmse(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        precision: int = 2
        ) -> float:
    rmse = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmse, precision)

#### Function for the model training part

def build_model(data: pd.DataFrame) -> dict[str, str]:
    df = drop_duplicates(data)
    X_train, X_test,y_train, y_test = dataset_train_test_split(df)
    final_train_df = preprocessor_func(X_train)
    final_test_df = preprocessor_func(X_test)
    y_pred = loaded_model.predict(final_test_df)
    y_pred[y_pred < 0] = 0
    rmse = compute_rmse(y_test, y_pred)
    return {'rmse': rmse}