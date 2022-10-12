import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from utils.client_utils import get_s3_object
from utils.common_utils import create_logger


def get_feature_names(df: pd.DataFrame, textual_feature_names: list, label_name: str):
    """
        Extracts numeric, categorical, text, date and label columns
        :param df: dataset
        :param textual_feature_names: plain text features
        :param label_name: label column name
        :return: numeric, categorical, text, date and label columns names
    """
    num_columns = df.select_dtypes(include=np.number).columns.tolist()
    numerical_feature_names = [i for i in num_columns if i != label_name]

    cat_columns = df.select_dtypes(include='object').columns.tolist()
    categorical_feature_names = [i for i in cat_columns if i not in textual_feature_names + [label_name]]

    date_columns = df.select_dtypes(include=np.datetime64).columns.tolist()
    date_feature_names = [i for i in date_columns]

    return numerical_feature_names, categorical_feature_names, textual_feature_names, date_feature_names, label_name


def remove_nan_features(dataset: pd.DataFrame, cutoff_threshold: float):
    """
        Remove features with multiple NAN values
        :param dataset: dataset to be visualized
        :param cutoff_threshold: maximum acceptable Nan values ration
        :return: dataset after the features removal
    """

    dataset_size = dataset.shape[0]
    drop_features = []
    for feature in dataset.columns:
        if dataset[feature].isna().sum() / dataset_size > cutoff_threshold:
            drop_features.append(feature)
            print("Feaure {} will be dropped!".format(feature))

    return dataset.drop(drop_features, axis=1)


def convert_data_type(dataset: pd.DataFrame, feature: str, function: any) -> pd.Series:
    """
        Applies the selected transformation to a dataframe column
        :param dataset: dataset
        :param feature: column name
        :param function: transformation function
        :return: transformed column as pd.Series
    """

    print("first type %s: " % feature, dataset[feature].dtype)
    feat = dataset[feature].apply(function)
    print("2nd type %s: " % feature, feat.dtype)
    return feat


def categorical_transformer(data: pd.DataFrame, feature: str, fill_value='Unknown') -> Tuple[
    pd.DataFrame, OneHotEncoder]:
    """
        Transforms numeric data
        :param data: dataset
        :param feature: numeric feature
        :param fill_value: value to replace all occurrences of missing_values
        :return: transformed data
    """

    filled = data[[feature]].fillna(fill_value).astype(str)
    encoder = OneHotEncoder(handle_unknown="ignore")
    transformed_data = encoder.fit_transform(filled)

    return pd.DataFrame(transformed_data.todense(),
                        columns=[f"{feature.lower().replace(' ', '_')}_{str(key)}" for key in
                                 range(transformed_data.shape[1])]), encoder


def numerical_transformer(data: pd.DataFrame, feature: str, fill_value=0.0) -> Tuple[list, MinMaxScaler]:
    """
        Transforms numeric data
        :param data: dataset
        :param feature: numeric feature,
        :param fill_value: value to replace all occurrences of missing_values
        :return: transformed data
    """
    filled = data[[feature]].fillna(fill_value)
    scaler = MinMaxScaler()
    transformed_data = scaler.fit_transform(filled)

    return transformed_data, scaler


def handle_dates(data: pd.DataFrame, feature: str, date_format='%Y-%m-%d', datetime=False) -> pd.DataFrame:
    """
        Splits date features to numeric
        :param data: dataset
        :param feature: the date column to be split
        :param date_format: date feature format
        :param datetime: True if hour, minute, second, microsecond should be extracted
        :return: dataFrame of the corresponding features
    """
    date_features = []
    for index, value in data[feature].iteritems():
        try:
            date_time_obj = dt.datetime.strptime(str(value), date_format)
        except ValueError:
            date_time_obj = dt.datetime.strptime(str(value), '%Y-%m-%d %H:%M:%S')

        year = date_time_obj.year
        week = date_time_obj.isocalendar()[1]
        features_dict = {f'{feature}_year': year, f'{feature}_month': date_time_obj.month,
                              f'{feature}_week': week, f'{feature}_day': date_time_obj.day,
                              f'{feature}_day_of_week': date_time_obj.weekday(),
                              # f'{feature}_week_of_year': f"{year}_{week}",
                              f'{feature}_day_of_year': date_time_obj.timetuple().tm_yday}
        if datetime:
            features_dict[f'{feature}hour'] = date_time_obj.hour
            features_dict[f'{feature}_minute'] = date_time_obj.minute
            features_dict[f'{feature}_second'] = date_time_obj.second
            features_dict[f'{feature}_microsecond'] = date_time_obj.microsecond

        date_features.append(features_dict)

    return pd.DataFrame(date_features)


def create_dataset(df: pd.DataFrame):
    """
        Convert DataFrame to tensors
        :param df: dataset
        :return: transformed data
    """

    sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features


def transform_prediction_dataset(dataset: pd.DataFrame, bucket: str, results_dir: str, selected_features: str,
                                 datetime_features: str, date_features: str, logger=None):
    if logger is None:
        logger = create_logger(__name__)

    selected_features = selected_features.split(',') if len(selected_features) > 0 else []
    datetime_features = datetime_features.split(',') if len(datetime_features) > 0 else []
    date_features = date_features.split(',') if len(date_features) > 0 else []
    dt_features = date_features + datetime_features

    df = dataset[selected_features]
    for datetime_feature in datetime_features:
        if datetime_feature in selected_features:
            df[datetime_feature] = pd.to_datetime(df[datetime_feature], format='%Y-%m-%d %H:%M:%S.%f')
    for date_feature in date_features:
        if date_feature in selected_features:
            df[date_feature] = pd.to_datetime(df[date_feature], format='%Y-%m-%d')

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [i for i in df.select_dtypes(include='object').columns.tolist() if i not in dt_features]

    for feature in datetime_features:
        if feature not in selected_features:
            continue
        logger.info(f"Starting transforming datetime feature {feature}")
        transformed_data = handle_dates(df, feature, '%Y-%m-%d %H:%M:%S.%f', True)
        numerical_cols.extend(transformed_data.columns.tolist())
        df = pd.concat([df.drop([feature], axis=1), transformed_data], axis=1)
        logger.info(f"Finished transforming datetime feature {feature}")

    for feature in date_features:
        if feature not in selected_features:
            continue
        logger.info(f"Starting transforming date feature {feature}")
        transformed_data = handle_dates(df, feature)
        numerical_cols.extend(transformed_data.columns.tolist())
        df = pd.concat([df.drop([feature], axis=1), transformed_data], axis=1)
        logger.info(f"Finished transforming date feature {feature}")

    for feature in numerical_cols:
        if feature not in selected_features:
            continue
        logger.info(f"Start processing categorical feature: {feature}")
        transformer = get_s3_object(bucket, f"{results_dir}/transformers", f"{feature}.dict", "dict")
        filled = df[[feature]].fillna(0.0)
        transformed_data = transformer.transform(filled)
        df[feature] = pd.Series(transformed_data.flatten())
        logger.info(f"Finished transforming numeric feature {feature}")

    for feature in cat_cols:
        if feature not in selected_features:
            continue
        logger.info(f"Start processing categorical feature: {feature}")
        transformer = get_s3_object(bucket, f"{results_dir}/transformers", f"{feature}.dict", "dict")
        filled = df[[feature]].fillna('Unknown').astype(str)
        transformed_data = transformer.transform(filled)
        transformed_df = pd.DataFrame(transformed_data.todense(),
                                      columns=[f"{feature.lower().replace(' ', '_')}_{str(key)}" for key in
                                               range(transformed_data.shape[1])])
        numerical_cols.extend(transformed_df.columns.tolist())
        df = pd.concat([df.drop([feature], axis=1), transformed_df], axis=1)
        logger.info(f"Finished processing categorical feature: {feature}")

    return df.astype(np.float32).to_numpy().tolist()
