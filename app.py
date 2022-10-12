import json

import numpy as np
import torch
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.serializers import JSONSerializer

from utils.anomaly_detection_utils import delete_outdated_endpoints, delete_outdated_models, delete_outdated_artifacts, \
    delete_outdated_endpoint_configs
from utils.client_utils import get_s3_objects, upload_file, get_execution_role, get_s3_object, get_boto3_client
from utils.common_utils import get_environmental_variable
from utils.constants import RANDOM_SEED
from utils.preprocess_utils import transform_prediction_dataset


def delete_endpoints(event, context):
    client = get_boto3_client('sagemaker')

    delete_outdated_endpoints(client)
    delete_outdated_endpoint_configs(client)

    return event


def delete_models(event, context):
    client = get_boto3_client('sagemaker')

    delete_outdated_models(client)

    return event


def delete_artifacts(event, context):
    client = get_boto3_client('sagemaker')

    delete_outdated_artifacts(client)

    return event


def train(event, context):

    # SETUP ENVIRONMENT
    np.random.seed(RANDOM_SEED)
    role = get_environmental_variable('EXECUTION_ROLE', 'SagemakerRole-Basic')  # 'AmazonSageMaker-ExecutionRole-20220105T144629'
    bucket = get_environmental_variable('S3_BUCKET_NAME', 'pc14-automated-pipeline')
    preprocess_output_folder = get_environmental_variable('PREPROCESS_S3_TARGET_FOLDER', 'csv_raw_data'
                                                                                         '/pacs_008_processed')
    anomaly_detection_dir = get_environmental_variable('ANOMALY_DETECTION_TARGET_FOLDER', 'anomaly_detection')
    trash_features = get_environmental_variable('TRASH_FEATURES', 'message_id,'
                                                                  'payment_identification_transaction_identification')
    file_type = get_environmental_variable('FILE_TYPE', 'csv')
    datetime_features = get_environmental_variable('DATETIME_FEATURES', 'creation_date_time')
    date_features = get_environmental_variable('DATE_FEATURES', 'interbank_settlement_date')

    df = get_s3_objects(bucket, preprocess_output_folder, file_type)
    training_uri = upload_file(df, bucket, f'{anomaly_detection_dir}/training', 'df_train', file_type)
    model_output_path = f"s3://{bucket}/{anomaly_detection_dir}/model"
    execution_role = get_execution_role(role, ['AmazonSageMakerFullAccess', 'AmazonS3FullAccess'])
    # Train my estimator
    pytorch_estimator = PyTorch(entry_point='train.py',
                                role=execution_role,
                                instance_type='ml.g4dn.xlarge',
                                instance_count=1,
                                framework_version='1.10.0',
                                py_version='py38',
                                output_path=model_output_path,
                                source_dir="code",
                                dependencies=['../utils', '../models'],
                                hyperparameters={
                                    'epochs': 15,
                                    'batch-size': 64,
                                    'learning-rate': 1e-3,
                                    'use-cuda': torch.cuda.is_available(),
                                    'trash-features': trash_features,
                                    'datetime-features': datetime_features,
                                    'date-features': date_features,
                                    'bucket': bucket,
                                    'results-dir': f'{anomaly_detection_dir}/results'
                                })
    pytorch_estimator.fit({'train': training_uri})

    return {
        "statusCode": 200,
        "body": json.dumps({
            "model-output-path": f'{anomaly_detection_dir}/model/{pytorch_estimator.base_job_name}/output',
            "execution-role": execution_role,
            "bucket": bucket,
            "results-dir": f'{anomaly_detection_dir}/results'
        }),
    }


def deploy(event, context):
    datetime_features = get_environmental_variable('DATETIME_FEATURES', 'creation_date_time')
    date_features = get_environmental_variable('DATE_FEATURES', 'interbank_settlement_date')
    elastic_inference = get_environmental_variable('ELASTIC_INFERENCE', 'True')

    resource_id_map = event['body']
    model_output_path = resource_id_map.get('model-output-path')
    role = resource_id_map.get('execution-role')
    bucket = resource_id_map.get('bucket')
    results_dir = resource_id_map.get('results-dir')
    metadata_df = get_s3_object(bucket, results_dir, 'metadata.csv', 'csv')
    threshold, max_loss, seq_len, n_features, embedding_dim, selected_features = metadata_df.iloc[0].tolist()
    selected_features = selected_features.replace('\'', '').replace('[', '').replace(']', '').replace(' ', '')

    model = PyTorchModel(
        entry_point="inference.py",
        source_dir="code",
        role=role,
        model_data=f'{model_output_path}/model.tar.gz',
        framework_version='1.10.0',
        py_version='py38',
        env={
            'ELASTIC_INFERENCE': elastic_inference,
            'SEQUENCE_LENGTH': str(seq_len),
            'N_FEATURES': str(n_features),
            'EMBEDDING_DIM': str(embedding_dim),
            'THRESHOLD': str(threshold)
        }
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.c4.xlarge",
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "endpoint-name": predictor.endpoint_name,
        }),
    }


def inference(event, context):
    bucket = get_environmental_variable('S3_BUCKET_NAME', 'pc14-automated-pipeline')
    preprocess_output_folder = get_environmental_variable('PREPROCESS_S3_TARGET_FOLDER', 'csv_raw_data'
                                                                                         '/pacs_008_processed')
    anomaly_detection_dir = get_environmental_variable('ANOMALY_DETECTION_TARGET_FOLDER', 'anomaly_detection')
    file_type = get_environmental_variable('FILE_TYPE', 'csv')
    datetime_features = get_environmental_variable('DATETIME_FEATURES', 'creation_date_time')
    date_features = get_environmental_variable('DATE_FEATURES', 'interbank_settlement_date')

    results_dir = f"{anomaly_detection_dir}/results"
    metadata_df = get_s3_object(bucket, results_dir, 'metadata.csv', file_type)
    threshold, max_loss, seq_len, n_features, embedding_dim, selected_features = metadata_df.iloc[0].tolist()
    selected_features = selected_features.replace('\'', '').replace('[', '').replace(']', '').replace(' ', '')

    client = get_boto3_client('sagemaker')
    df = get_s3_objects(bucket, preprocess_output_folder, file_type)
    data = transform_prediction_dataset(df, bucket, results_dir, selected_features, datetime_features,
                                        date_features)
