import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


useful_features = ['Foundation', 'KitchenQual',
                   'TotRmsAbvGrd', 'WoodDeckSF',
                   'YrSold', '1stFlrSF']


# Standard scaling function

def scaler_func(data: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    continuous_columns = (data[useful_features]
                          .select_dtypes(include='number')
                          .columns)
    scaled_columns = scaler.transform(data[continuous_columns])
    continuous_features_df = pd.DataFrame(data=scaled_columns,
                                          columns=continuous_columns)
    return continuous_features_df

# One hot encoding function


def encoder_func(data: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    categorical_columns = (data[useful_features]
                           .select_dtypes(include='object')
                           .columns)
    categorical_columns_list = categorical_columns.tolist()
    labels = encoder.get_feature_names_out(categorical_columns_list)
    encoded_data = encoder.transform(data[categorical_columns_list]).toarray()
    categorical_features_df = pd.DataFrame(data=encoded_data, columns=labels)
    return categorical_features_df

# Function to join continuous scaled and categorical scaled datasets


def join_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    final_df = df1.join(df2)
    return final_df
