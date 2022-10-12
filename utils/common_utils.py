import datetime
import logging
import os
import sys


def get_environmental_variable(name: str, default_value=None, is_numeric=False, is_list=False, is_bool=False):
    value = os.getenv(name, default_value)
    value = default_value if default_value is not None and value == '' else value
    if value is None or value == '':
        return None
    else:
        if is_list:
            if value == '':
                value = []
            else:
                value = [float(el.strip()) if is_numeric else el.strip() for el in str(value).strip().split(',')]
        else:
            value = float(value.strip()) if is_numeric else value.strip().lower() == 'true' if is_bool else value.strip()
    return value


def date_serializer(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()


def create_logger(name: str):
    logger = logging.getLogger(name)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    return logger
