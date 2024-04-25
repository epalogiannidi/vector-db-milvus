import logging
import os
import sys
import time
from logging import handlers
from typing import Dict

import yaml  # type: ignore


def set_up_logger():
    logdir = "logs"
    logname = time.strftime("%m%d%Y-%H:%M:%S")

    os.makedirs(logdir, exist_ok=True)

    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    logger.addHandler(ch)

    fh = handlers.RotatingFileHandler(
        os.path.join(logdir, logname), maxBytes=(1048576 * 5), backupCount=7
    )
    fh.setFormatter(format)
    logger.addHandler(fh)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    return logger


def load_config(config_path: str) -> Dict:
    """
     Loads a yaml file that contains all the information about the trained model,
     as it is saved on AWS custom labels (AWS Rekognition)

     :param config_path: str
         The path to the configuration file

    :return: Dict[str, Any]
     The dictionary with the trained model information

    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError
    else:
        print("Model info file loaded successfully.")
    return config
