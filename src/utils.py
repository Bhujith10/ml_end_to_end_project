import os
import sys
import pickle
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold

def save_object(file_path, obj, context):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logging.info(f'successfully saved {context} object')
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,x_test,y_train,y_test,models,params):
    model_report = {}
    try:
        for model_name,model in models.items():
            cv = KFold(n_splits=4, shuffle=True, random_state=0)
            gs = GridSearchCV(model,param_grid=params[model_name],cv=cv).fit(x_train,y_train)
            print(model_name,' ',gs.cv_results_)
            
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            pred_test = model.predict(x_test)
            r2 = r2_score(y_test,pred_test)

            model_report[model_name] = r2
            logging.info('Completed model building and training')
        return model_report
    except Exception as e:
        raise CustomException(e,sys)