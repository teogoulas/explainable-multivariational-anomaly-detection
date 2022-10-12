import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from models.deep_learning_model.Dataset import CustomDataset
from models.deep_learning_model.VariationalAutoencoder import VAE
from utils.client_utils import upload_file
from utils.preprocess_utils import remove_nan_features, handle_dates, numerical_transformer, categorical_transformer

if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()

    # hyper-parameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--use-cuda', type=bool, default=False)
    parser.add_argument('--elastic-inference', type=bool, default=False)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument('--selected-features', type=str,
                        default='creation_date_time,debtor_account_iban,interbank_settlement_amount_value,debtor_agent_financial_institution_identification_bicfi,creditor_agent_financial_institution_identification_bicfi')
    parser.add_argument('--trash-features', type=str, default='message_id,payment_identification_transaction_identification')
    parser.add_argument('--datetime-features', type=str, default='creation_date_time')
    parser.add_argument('--date-features', type=str, default='interbank_settlement_date')
    parser.add_argument('--bucket', type=str, default='pc14-automated-pipeline')
    parser.add_argument('--results-dir', type=str, default='anomaly_detection/results')

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    # parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()
    print("Received arguments {}".format(args))

    np.random.seed(args.random_seed)
    epochs = args.epochs
    batch_size = args.batch_size
    split_ratio = args.train_test_split_ratio
    embedding_dim = args.embedding_dim
    bucket = args.bucket
    output_data_dir = args.output_data_dir
    model_dir = args.model_dir
    results_dir = args.results_dir
    elastic_inference = args.elastic_inference
    device = torch.device("cuda" if args.use_cuda and not elastic_inference else "cpu")

    selected_features = args.selected_features.split(',') if len(args.selected_features) > 0 else []
    trash_features = args.trash_features.split(',') if len(args.trash_features) > 0 else []
    date_features = args.date_features.split(',') if len(args.date_features) > 0 else []
    datetime_features = args.datetime_features.split(',') if len(args.datetime_features) > 0 else []
    dt_features = date_features + datetime_features

    input_data_path = f"{args.train}/df_train.csv"
    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(
        input_data_path
    )
    logger.info(f"Dataframe shape: {df.shape}")

    selected_features = [feat for feat in selected_features if feat not in trash_features]
    df = df[selected_features]
    df = remove_nan_features(df, 0.5)

    for datetime_feature in datetime_features:
        if datetime_feature in selected_features:
            df[datetime_feature] = pd.to_datetime(df[datetime_feature], format='%Y-%m-%d %H:%M:%S.%f')
    for date_feature in date_features:
        if date_feature in selected_features:
            df[date_feature] = pd.to_datetime(df[date_feature], format='%Y-%m-%d')

    training_dataset = df.copy()
    logger.info(f"Training dataframe shape: {training_dataset.shape}")

    numerical_cols = training_dataset.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [i for i in training_dataset.select_dtypes(include='object').columns.tolist() if i not in dt_features]

    transformers = {}
    for feature in datetime_features:
        if feature not in selected_features:
            continue
        logger.info(f"Starting transforming datetime feature {feature}")
        transformed_data = handle_dates(training_dataset, feature, '%Y-%m-%d %H:%M:%S.%f', True)
        numerical_cols.extend(transformed_data.columns.tolist())
        training_dataset = pd.concat([training_dataset.drop([feature], axis=1), transformed_data], axis=1)
        logger.info(f"Finished transforming datetime feature {feature}")
        logger.info(f"Training dataframe shape: {training_dataset.shape}")

    for feature in date_features:
        if feature not in selected_features:
            continue
        logger.info(f"Starting transforming date feature {feature}")
        transformed_data = handle_dates(training_dataset, feature)
        numerical_cols.extend(transformed_data.columns.tolist())
        training_dataset = pd.concat([training_dataset.drop([feature], axis=1), transformed_data], axis=1)
        logger.info(f"Finished transforming date feature {feature}")
        logger.info(f"Training dataframe shape: {training_dataset.shape}")

    for feature in numerical_cols:
        if feature not in selected_features:
            continue
        logger.info(f"Starting transforming numeric feature {feature}")
        transformed_data, transformer = numerical_transformer(training_dataset, feature)
        training_dataset[feature] = pd.Series(transformed_data.flatten())
        transformers[feature] = transformer
        logger.info(f"Finished transforming numeric feature {feature}")
        logger.info(f"Training dataframe shape: {training_dataset.shape}")

    for feature in cat_cols:
        if feature not in selected_features:
            continue
        logger.info(f"Starting transforming categorical feature {feature}")
        transformed_data, transformer = categorical_transformer(training_dataset, feature)
        numerical_cols.extend(transformed_data.columns.tolist())
        training_dataset = pd.concat([training_dataset.drop([feature], axis=1), transformed_data], axis=1)
        transformers[feature] = transformer
        logger.info(f"Finished transforming categorical feature {feature}")
        logger.info(f"Training dataframe shape: {training_dataset.shape}")

    upload_file(transformers, bucket, results_dir, "transformers", 'dict')
    upload_file(training_dataset, bucket, results_dir, "training_dataset", 'csv')

    train_df, val_df = train_test_split(
        training_dataset,
        test_size=split_ratio,
        random_state=args.random_seed
    )

    train_dataset = CustomDataset(train_df.values, train_df.shape[0])
    val_dataset = CustomDataset(val_df.values, val_df.shape[0])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = VAE(train_df.shape[1])
    model = model.to(device)

    trainer = pl.Trainer(gpus=args.gpus, max_epochs=epochs)
    trainer.fit(model, train_dataloader, val_dataloader)

    metadata_df = pd.DataFrame({
        'seq_len': [train_df.shape[1]],
        'selected_features': [selected_features]
    })
    upload_file(metadata_df, bucket, results_dir, 'metadata', 'csv')

    # ... train `model`, then save it to `model_dir`
    if elastic_inference:
        trace_input = torch.randn_like(next(iter(val_dataloader)))
        traced_model = torch.jit.trace(model.eval(), trace_input)
        model_dir = os.path.join(model_dir, "model.pt")
        torch.jit.save(traced_model, model_dir)
    else:
        with open(os.path.join(model_dir, "model.pth"), 'wb') as f:
            torch.save(model.state_dict(), f)
