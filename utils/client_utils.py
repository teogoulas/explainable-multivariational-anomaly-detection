import io
import json
import os
import pickle
import time
from datetime import datetime

import boto3
import botocore
import pandas as pd
import sagemaker
from boto3 import client


def get_s3_client() -> client:
    """
        Gets S3 client
        :return: boto3 client.
    """

    # AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

    s3_client = client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )

    return s3_client


def download_csv_file(s3_client: client, bucket: str, filepath: str, parse_dates: list, date_parser: any, sep=','):
    """
        Gets csv from S3 bucket
        :param s3_client: S3 client
        :param bucket: S3 bucket name
        :param filepath: object's filepath
        :param parse_dates: date fields (if any)
        :param date_parser: date fields parser function
        :param sep: csv seperator
        :return: specified object.
    """

    s3_object = None
    response = s3_client.get_object(Bucket=bucket, Key=filepath)

    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        s3_object = pd.read_csv(io.BytesIO(response['Body'].read()), encoding='utf-8', sep=sep, parse_dates=parse_dates,
                                date_parser=date_parser)
    else:
        print(f"Unsuccessful S3 get_object response. Status - {status}")

    return s3_object


def upload_csv(df: pd.DataFrame, bucket_name: str, filename: str, sep=',', header=True) -> str:
    """
        Saves dataframe as csv file to AWS S3 bucket
        :param df: data frame to be uploaded
        :param bucket_name: AWS S3 bucket name
        :param filename: csv file name
        :param sep: csv seperator
        :param header: if headers are included as csv header
        :return: file S3 URI
    """

    print("Ready to create csv file {}".format(filename))
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, header=header, sep=sep,
              encoding="utf-8")  # , quoting=csv.QUOTE_NONE, quotechar='', escapechar="\\"

    print("Ready to upload csv file {}".format(filename))
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).put_object(Key=filename, Body=csv_buffer.getvalue(), ContentType='text/csv; charset=utf-8')
    print("File {0} successfully uploaded to S3 bucket {1}".format(filename, bucket_name))
    return f"s3://{bucket_name}/{filename}"


def check_bucket_permission(bucket: str) -> bool:
    """
        Checks if S3 bucket exists and the spesified user has access to it
        :param bucket: S3 bucket name
        :return: True if the user has access to the S3 bucket.
    """
    permission = False
    try:
        boto3.Session().client("s3").head_bucket(Bucket=bucket)
    except botocore.exceptions.ParamValidationError as e:
        print(
            "Hey! You either forgot to specify your S3 bucket or you gave your bucket an invalid name!"
        )
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "403":
            print(f"Hey! You don't have permission to access the bucket, {bucket}.")
        elif e.response["Error"]["Code"] == "404":
            print(f"Hey! Your bucket, {bucket}, doesn't exist!")
        else:
            raise
    else:
        print(f"Access to s3://{bucket} is confirmed!")
        permission = True
    return permission


def download_excel_file(bucket: str, filepath: str):
    """
        Downloads an Excel (xlsx) file from S3 bucket
        :param bucket: S3 bucket name
        :param filepath: object's filepath
        :return: specified object.
    """

    s3_object = None
    response = get_s3_client().get_object(Bucket=bucket, Key=filepath)

    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        s3_object = pd.read_excel(io.BytesIO(response['Body'].read()), engine='openpyxl')
    else:
        print(f"Unsuccessful S3 get_object response. Status - {status}")

    return s3_object


def download_pickle_file(s3_client: client, bucket: str, filepath: str):
    """
        Downloads an dict file from S3 bucket
        :param s3_client: S3 client
        :param bucket: S3 bucket name
        :param filepath: object's filepath
        :return: specified object.
    """

    s3_object = None
    response = s3_client.get_object(Bucket=bucket, Key=filepath)

    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        s3_object = pickle.loads(response['Body'].read())
    else:
        print(f"Unsuccessful S3 get_object response. Status - {status}")

    return s3_object


def upload_excel(df: pd.DataFrame, bucket_name: str, filename: str) -> str:
    """
        Saves dataframe as xlsx file to AWS S3 bucket
        :param df: data frame to be uploaded
        :param bucket_name: AWS S3 bucket name
        :param filename: excel file name
        :return: file S3 URI
    """
    with io.BytesIO() as output:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer)
        data = output.getvalue()

    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).put_object(Key=filename, Body=data)
    print("File {0} successfully uploaded to S3 bucket {1}".format(filename, bucket_name))
    return f"s3://{bucket_name}/{filename}"


def upload_dict(dictionary: dict, bucket_name: str, filename: str) -> list:
    """
        Saves dictionary as file to AWS S3 bucket
        :param dictionary: dictionary to be uploaded
        :param bucket_name: AWS S3 bucket name
        :param filename: file name
        :return: file S3 URI
    """

    files = []
    for key in dictionary.keys():
        data = pickle.dumps(dictionary[key])

        s3 = boto3.resource('s3')
        s3.Bucket(bucket_name).put_object(Key=f"{filename}/{key}.dict", Body=data)
        print("File {0} successfully uploaded to S3 bucket {1}".format(f"{key}.dict", f"{bucket_name}/{filename}"))
        files.append(f"s3://{bucket_name}/{filename}")
    return files


def upload_buffer(buffer: io.BytesIO, bucket_name: str, filename: str) -> str:
    """
        Saves dictionary as file to AWS S3 bucket
        :param buffer: buffer to be uploaded
        :param bucket_name: AWS S3 bucket name
        :param filename: file name
        :return: file S3 URI
    """

    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).put_object(Key=filename, Body=buffer.getvalue())
    print("File {0} successfully uploaded to S3 bucket {1}".format(filename, bucket_name))
    return f"s3://{bucket_name}/{filename}"


def upload_json(json_object, bucket_name: str, filename: str) -> str:
    """
        Saves json as file to AWS S3 bucket
        :param json_object: json object to be uploaded
        :param bucket_name: AWS S3 bucket name
        :param filename: file name
        :return: file S3 URI
    """

    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).put_object(Key=filename, Body=json.dumps(json_object))
    print("File {0} successfully uploaded to S3 bucket {1}".format(filename, bucket_name))
    return f"s3://{bucket_name}/{filename}"


def get_s3_objects(bucket: str, prefix: str, default_file_type: str, parse_dates=None, date_format='%Y-%m-%d %H:%M:%S',
                   sep=','):
    """
        Gets all objects from S3 bucket
        :param bucket: S3 bucket name
        :param prefix: S3 sub-folder name
        :param default_file_type: objects default file type (xlsx, xls, csv)
        :param parse_dates: date fields (if any)
        :param date_format: date fields format
        :param sep: csv seperator
        :return: dataframe containing records from all objects.
    """

    objects = get_s3_client().list_objects_v2(Bucket=bucket, Prefix=prefix)
    dataset = pd.DataFrame()
    for obj in objects['Contents']:
        key = obj['Key']
        slash_index = key.rfind('/')
        filepath = key[0:slash_index]
        filename = key[slash_index + 1:]
        if filename == '':
            continue
        filename = filename.strip().split('.')
        if len(filename) > 1:
            file_type = filename[1]
            filename = f"{filename[0]}.{file_type}"
        else:
            file_type = default_file_type
            filename = filename[0]
        df = get_s3_object(bucket, filepath, filename, file_type, parse_dates, date_format, sep)
        if df.shape[0] > 0:
            # df = df.reindex(sorted(df.columns), axis=1)
            dataset = pd.concat([dataset, df], axis=0) if dataset.shape[0] > 0 else df

    return dataset


def get_s3_object(bucket: str, prefix: str, filename: str, file_type='csv', parse_dates=None,
                  date_format='%Y-%m-%d %H:%M:%S', sep=','):
    """
        Gets object from S3 bucket
        :param bucket: S3 bucket name
        :param prefix: S3 sub-folder name
        :param filename: file name
        :param file_type: object file type (xlsx, xls, csv)
        :param parse_dates: date fields (if any)
        :param date_format: date fields format
        :param sep: csv seperator
        :return: specified object.
    """

    if parse_dates is None:
        date_parser = None
        parse_dates = []
    else:
        date_parser = lambda x: datetime.strptime(x, date_format)

    filepath = f"{prefix}/{filename}" if len(prefix) > 0 else f"{filename}"

    if check_bucket_permission(bucket):
        if file_type in ['xls', 'xlsx']:
            dataset = download_excel_file(bucket, filepath)
        elif file_type == 'csv':
            dataset = download_csv_file(get_s3_client(), bucket, filepath, parse_dates, date_parser, sep)
        else:
            dataset = download_pickle_file(get_s3_client(), bucket, filepath)
    else:
        dataset = pd.DataFrame()
    print(
        f'Dictionary successfully downloaded!' if file_type == 'dict' else f'The shape of the dataset is: {dataset.shape}')
    return dataset


def upload_file(data, bucket: str, prefix: str, filename: str, file_type: str, sep=',', header=True) -> str:
    """
        Saves file to AWS S3 bucket
        :param data: data to be uploaded
        :param bucket: S3 bucket name
        :param prefix: S3 sub-folder name
        :param filename: file name
        :param file_type: object file type (xlsx, xls, csv, dict)
        :param sep: csv seperator
        :param header: if headers are included as csv header
        :return: file S3 URI
    """

    filepath = f"{prefix}/{filename}.{file_type}" if len(prefix) > 0 else f"{filename}.{file_type}"

    uri = ''
    if file_type in ['xls', 'xlsx']:
        uri = upload_excel(data, bucket, filepath)
    elif file_type == 'csv':
        uri = upload_csv(data, bucket, filepath, sep, header)
    elif file_type == 'dict':
        uri = upload_dict(data, bucket, f"{prefix}/{filename}")
    elif file_type == 'pth':
        uri = upload_buffer(data, bucket, filepath)
    elif file_type == 'json':
        uri = upload_json(data, bucket, filepath)
    return uri


def get_execution_role(execution_role: str, policies: list, service='sagemaker') -> str:
    """
        Gets Sagemaker execution role
        :return: Sagemaker execution role.
    """

    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        role = get_or_create_iam_role(execution_role, policies, service)

    print(f"Role {role}")
    return role


def get_or_create_iam_role(role_name: str, policies: list, service='sagemaker'):
    iam = boto3.client("iam")

    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": f"{service}.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }

    try:
        create_role_response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document)
        )
        role_arn = create_role_response["Role"]["Arn"]
        print("Created", role_arn)

        print("Attaching policies...")
        for policy in policies:
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=f"arn:aws:iam::aws:policy/{policy}"
            )

        print("Waiting for a minute to allow IAM role policy attachment to propagate")
        time.sleep(60)
    except iam.exceptions.EntityAlreadyExistsException:
        print("The role " + role_name + " exists, ignore to create it")
        role_arn = boto3.resource('iam').Role(role_name).arn

    print("Done.")
    return role_arn


def delete_iam_role(role_name):
    iam = boto3.client("iam")
    iam.detach_role_policy(PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess", RoleName=role_name)
    iam.detach_role_policy(PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess", RoleName=role_name)
    iam.delete_role(RoleName=role_name)


def get_boto3_client(service_name: str) -> boto3.Session.client:
    region = boto3.Session().region_name
    session = boto3.Session(region_name=region)
    return session.client(service_name=service_name)
