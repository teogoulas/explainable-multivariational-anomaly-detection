import boto3
from sagemaker import Predictor, Session
from sagemaker.deserializers import BytesDeserializer
from sagemaker.serializers import JSONSerializer

from models.Exceptions.ResourceError import ResourceError
from utils.client_utils import get_s3_object, get_s3_objects
from utils.preprocess_utils import transform_prediction_dataset


def list_endpoints(sagemaker: boto3.client, status=None, sort_by='CreationTime', order='Descending') -> list:
    if status is not None:
        callback = lambda token: sagemaker.list_endpoints(SortBy=sort_by, SortOrder=order,
                                                          StatusEquals=status) if token is None else sagemaker.list_endpoints(
            SortBy=sort_by, SortOrder=order, StatusEquals=status, NextToken=token)
    else:
        callback = lambda token: sagemaker.list_endpoints(SortBy=sort_by,
                                                          SortOrder=order) if token is None else sagemaker.list_endpoints(
            SortBy=sort_by, SortOrder=order, NextToken=token)

    list_response = callback(None)

    endpoints = list_response['Endpoints']

    if 'NextToken' in list_response.keys():
        next_token = list_response['NextToken']
        while True:
            list_response = callback(next_token)
            endpoints += list_response['Endpoints']
            if 'NextToken' in list_response.keys():
                next_token = list_response['NextToken']
            else:
                break

    return [endpoint['EndpointName'] for endpoint in endpoints]


def list_endpoint_configs(sagemaker: boto3.client, sort_by='CreationTime', order='Descending') -> list:
    list_response = sagemaker.list_endpoint_configs(SortBy=sort_by, SortOrder=order)

    endpoints = list_response['EndpointConfigs']

    if 'NextToken' in list_response.keys():
        next_token = list_response['NextToken']
        while True:
            list_response = sagemaker.list_endpoint_configs(SortBy=sort_by, SortOrder=order, NextToken=next_token)
            endpoints += list_response['EndpointConfigs']
            if 'NextToken' in list_response.keys():
                next_token = list_response['NextToken']
            else:
                break

    return [endpoint['EndpointConfigName'] for endpoint in endpoints]


def get_latest_endpoint(sagemaker: boto3.client) -> str:
    endpoints = list_endpoints(sagemaker, 'InService')

    if len(endpoints) == 0:
        raise ResourceError("NoEndpointFoundException", "Get latest endpoint", "Cannot find any endpoint")

    return endpoints[0]


def delete_outdated_endpoints(sagemaker: boto3.client):
    endpoints = list_endpoints(sagemaker)
    for endpoint in endpoints:
        try:
            sagemaker.delete_endpoint(EndpointName=endpoint)
        except Exception as e:
            error_code = e.response['Error']['Code']
            print(f"Failed to delete endpoint {endpoint}, due to {error_code}")

    endpoints = list_endpoints(sagemaker, 'InService')
    if len(endpoints) > 0:
        raise ResourceError('ResourceCleanupInProgressException', 'Endpoints cleanup',
                            f"Outdated endpoints cleanup failed with outdated endpoints [{', '.join(endpoints)}]")

    print("Successfully cleanup outdated endpoints.")


def delete_outdated_endpoint_configs(sagemaker: boto3.client):
    configs = list_endpoint_configs(sagemaker)
    for config in configs:
        try:
            sagemaker.delete_endpoint_config(EndpointConfigName=config)
        except Exception as e:
            error_code = e.response['Error']['Code']
            print(f"Failed to delete endpoint config {config}, due to {error_code}")

    configs = list_endpoint_configs(sagemaker)
    if len(configs) > 0:
        raise ResourceError('ResourceCleanupInProgressException', 'Endpoint configurations cleanup',
                            f"Outdated endpoint configurations cleanup failed with outdated endpoint configurations [{', '.join(configs)}]")

    print("Successfully cleanup outdated endpoint configurations.")


def list_models(sagemaker: boto3.client, sort_by='CreationTime', order='Descending') -> list:
    list_response = sagemaker.list_models(
        SortBy=sort_by,
        SortOrder=order
    )

    models = list_response['Models']

    if 'NextToken' in list_response.keys():
        next_token = list_response['NextToken']
        while True:
            list_response = sagemaker.list_models(
                SortBy=sort_by,
                SortOrder=order,
                NextToken=next_token
            )
            models += list_response['Models']
            if 'NextToken' in list_response.keys():
                next_token = list_response['NextToken']
            else:
                break

    return [model['ModelName'] for model in models]


def delete_outdated_models(sagemaker: boto3.client):
    models = list_models(sagemaker)
    for model in models:
        try:
            sagemaker.delete_model(ModelName=model)
        except Exception as e:
            error_code = e.response['Error']['Code']
            print(f"Failed to delete model {model}, due to {error_code}")

    models = list_models(sagemaker)
    if len(models) > 0:
        raise ResourceError('ResourceCleanupInProgressException', 'Models cleanup',
                            f"Outdated models cleanup failed with outdated models [{', '.join(models)}]")

    print("Successfully cleanup outdated models.")


def list_artifacts(sagemaker: boto3.client, sort_by='CreationTime', order='Descending') -> list:
    list_response = sagemaker.list_artifacts(
        SortBy=sort_by,
        SortOrder=order
    )

    artifacts = list_response['ArtifactSummaries']

    if 'NextToken' in list_response.keys():
        next_token = list_response['NextToken']
        while True:
            list_response = sagemaker.list_artifacts(
                SortBy=sort_by,
                SortOrder=order,
                NextToken=next_token
            )
            artifacts += list_response['ArtifactSummaries']
            if 'NextToken' in list_response.keys():
                next_token = list_response['NextToken']
            else:
                break

    return artifacts


def list_associations(sagemaker: boto3.client, source_arn: str):
    list_response = sagemaker.list_associations(SourceArn=source_arn)

    associations = list_response['AssociationSummaries']

    if 'NextToken' in list_response.keys():
        next_token = list_response['NextToken']
        while True:
            list_response = sagemaker.list_associations(
                SourceArn=source_arn,
                NextToken=next_token
            )
            associations += list_response['AssociationSummaries']
            if 'NextToken' in list_response.keys():
                next_token = list_response['NextToken']
            else:
                break

    return [association['DestinationArn'] for association in associations]


def delete_outdated_associations(sagemaker: boto3.client, source_arn: str):
    associations = list_associations(sagemaker, source_arn)

    for association in associations:
        try:
            sagemaker.delete_association(SourceArn=source_arn, DestinationArn=association)
        except Exception as e:
            error_code = e.response['Error']['Code']
            print(f"Failed to delete association {association}, due to {error_code}")

    associations = list_associations(sagemaker, source_arn)
    if len(associations) > 0:
        raise ResourceError('ResourceCleanupInProgressException', 'Associations cleanup',
                            f"Outdated associations cleanup failed with outdated associations [{', '.join(associations)}]")

    print("Successfully cleanup outdated associations.")


def delete_outdated_artifacts(sagemaker: boto3.client):
    artifacts = list_artifacts(sagemaker)
    for artifact in artifacts:
        try:
            delete_outdated_associations(sagemaker, artifact['ArtifactArn'])
            sagemaker.delete_artifact(ArtifactArn=artifact['ArtifactArn'], Source=artifact['Source'])
        except Exception as e:
            error_code = e.response['Error']['Code']
            print(f"Failed to delete model {artifact}, due to {error_code}")

    artifacts = list_artifacts(sagemaker)
    if len(artifacts) > 0:
        # raise ResourceError('ResourceCleanupInProgressException', 'Artifacts cleanup',
        #                     f"Outdated artifacts cleanup failed with outdated artifacts [{', '.join(artifacts)}]")
        print(f"Outdated artifacts cleanup failed with outdated artifacts [{', '.join(artifacts)}]")

    print("Successfully cleanup outdated artifacts.")


def invoke_endpoint(sagemaker: boto3.client, bucket: str, anomaly_detection_dir: str, preprocess_output_folder: str,
                    datetime_features: str, date_features: str, file_type='csv'):
    results_dir = f"{anomaly_detection_dir}/results"
    metadata_df = get_s3_object(bucket, results_dir, 'metadata.csv', file_type)
    threshold, max_loss, seq_len, n_features, embedding_dim, selected_features = metadata_df.iloc[0].tolist()
    selected_features = selected_features.replace('\'', '').replace('[', '').replace(']', '').replace(' ', '')

    df = get_s3_objects(bucket, preprocess_output_folder, file_type)
    data = transform_prediction_dataset(df, bucket, results_dir, selected_features, datetime_features,
                                        date_features)

    endpoint_name = get_latest_endpoint(sagemaker)
    predictor = Predictor(
        endpoint_name,
        Session(),
        serializer=JSONSerializer(),
        deserializer=BytesDeserializer(),
    )
    res = predictor.predict(data)

    print(res)
