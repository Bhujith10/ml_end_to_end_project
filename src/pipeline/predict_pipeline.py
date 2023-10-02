import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            scaled_features = preprocessor.transform(features)
            pred = model.predict(scaled_features)
            return pred
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
     def __init__(self,gender:str,race_ethnicity:str,parental_education:str,lunch:str,test_preparation_course:str,
                  writing_score:int,reading_score:int):
          self.gender = gender
          self.race_ethnicity = race_ethnicity
          self.parental_education = parental_education
          self.lunch = lunch
          self.test_preparation_course = test_preparation_course
          self.reading_score = reading_score
          self.writing_score = writing_score

     def create_dataframe(self):
         try:
            custom_data_input_dict = {
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_education':[self.parental_education],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]
            }
            custom_datframe = pd.DataFrame(custom_data_input_dict)

            custom_datframe = custom_datframe.rename(columns={'reading_score':'reading score',
                                                                'writing_score':'writing score',
                                                                'race_ethnicity':'race/ethnicity',
                                                                'parental_education':'parental level of education',
                                                                'test_preparation_course':'test preparation course'})
            return custom_datframe
         except Exception as e:
             raise CustomException(e,sys)
     

