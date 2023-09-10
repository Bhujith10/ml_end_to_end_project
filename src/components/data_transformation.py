import sys
import os
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils import save_object
from data_ingestion import DataIngestion

class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
            numerical_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                     ('encoder', OneHotEncoder()),
                     ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info(f'Categorical Columns :{categorical_columns}')
            logging.info(f'Numerical Columns :{numerical_columns}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',numerical_pipeline,numerical_columns),
                    ('cat_pipeline',categorical_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            
            preprocessor = self.get_data_transformer_object()
            logging.info("Obtaining preprocessing object")

            target_column_name = 'math score'
            
            train_df_input = train_df.drop(target_column_name,axis=1)
            test_df_input = test_df.drop(target_column_name,axis=1)

            train_df_target = train_df[target_column_name]
            test_df_target = test_df[target_column_name]

            train_arr = preprocessor.fit_transform(train_df_input)
            test_arr = preprocessor.transform(test_df_input)

            train_arr = np.c_[train_arr, np.array(train_df_target)]
            test_arr = np.c_[test_arr, np.array(test_df_target)]
            logging.info("Completed preprocessing")

            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                        obj = preprocessor,
                        context = 'preprocessor')
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    
    obj1 = DataIngestion()
    train_path, test_path = obj1.initiate_data_ingestion()
    obj2 = DataTransformation()
    obj2.initiate_data_transformation(train_path, test_path)
