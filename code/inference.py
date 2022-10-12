from __future__ import absolute_import

import json
import logging
import os
import sys

import shap
import torch
from torch.utils.data import DataLoader

from models.deep_learning_model.Dataset import CustomDataset
from models.deep_learning_model.VariationalAutoencoder import VAE
from utils.client_utils import upload_file
from utils.common_utils import get_environmental_variable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def model_fn(model_dir):
    elastic_inference = get_environmental_variable('ELASTIC_INFERENCE', 'False', is_bool=True)
    seq_len = get_environmental_variable('SEQUENCE_LENGTH', is_numeric=True)
    try:
        if elastic_inference:
            loaded_model = torch.jit.load(os.path.join(model_dir, "model.pt"), map_location=torch.device("cpu"))
            return loaded_model
        else:
            model = VAE(int(seq_len))
            with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
                model.load_state_dict(torch.load(f))
            return model
    except Exception as e:
        logger.exception(f"Exception in model fn {e}")
        return None


def predict_fn(input_data, model):
    elastic_inference = get_environmental_variable('ELASTIC_INFERENCE', 'True', False, False, True)
    bucket = get_environmental_variable('BUCKET', 'pc14-automated-pipeline')
    logger.info(
        "Performing {} inference with{} input of size {}".format(
            'EIA' if elastic_inference else 'Standard',
            ' Torch JIT context with' if elastic_inference else '',
            next(iter(input_data)).shape
        )
    )
    # With EI, client instance should be CPU for cost-efficiency. Subgraphs with unsupported arguments run locally.
    # Server runs with CUDA
    device = torch.device("cuda" if torch.cuda.is_available() and not elastic_inference else "cpu")
    if not elastic_inference:
        model = model.to(device)
        model.eval()

    preds = {}
    with torch.no_grad():
        for _, batch in enumerate(input_data):
            res = model._shared_eval_step(batch)
            if 'kl' not in preds.keys():
                preds['kl'] = res['kl']
            else:
                preds['kl'] = torch.cat((preds['kl'], res['kl']))
            if 'reconstruction' not in preds.keys():
                preds['reconstruction'] = res['reconstruction']
            else:
                preds['reconstruction'] = torch.cat((preds['reconstruction'], res['reconstruction']))

    data_list = input_data.dataset.x
    explainer = shap.DeepExplainer(model, data_list)
    shap_values = explainer.shap_values(data_list)
    preds['shap_values'] = shap_values
    upload_file(preds, bucket, 'anomaly_detection/results/', 'predictions', 'dict')
    return {'statusCode': 200}


def input_fn(request_body, request_content_type):
    batch_size = get_environmental_variable('BATCH_SIZE', '64', is_numeric=True)
    print(f"Content Type: {request_content_type}")
    if request_content_type == "application/json":
        data = json.loads(request_body)
        dataset = CustomDataset(data, len(data))
        return DataLoader(dataset, batch_size=int(batch_size), shuffle=True, drop_last=True)
    else:
        raise Exception("Requested unsupported ContentType in Accept: " + request_content_type)


def output_fn(prediction, response_content_type):
    print("PREDICTION", prediction)

    if response_content_type == "application/json":
        return json.dumps(prediction), response_content_type

    raise Exception("Requested unsupported ContentType in Accept: " + response_content_type)
