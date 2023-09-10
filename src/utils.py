import os
import sys
import pickle
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj, context):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logging.info(f'successfully saved {context} object')
    except Exception as e:
        raise CustomException(e,sys)