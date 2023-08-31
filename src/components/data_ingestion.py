import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

"""
Dataclasses, as the name clearly suggests, are classes that are meant to hold data. 
The motivation behind this module is that we sometimes define classes that only act as data containers and when we do that, 
we spend a consequent amount of time writing boilerplate code with tons of arguments, 
an ugly __init__ method and many overridden functions.

Dataclasses alleviates this problem while providing additional useful methods under the hood. 

https://towardsdatascience.com/9-reasons-why-you-should-start-using-python-dataclasses-98271adadc66
"""

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.dataIngestionConfig = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered data ingestion step')
        try:
          df = pd.read_csv('notebook\StudentsPerformance.csv')  
          logging.info('Completed reading the dataset as a dataframe')

          ## Here os.path.dirname gets the directory name from the path specified.
          ## In this case the dirname is artifacts.
          ## Creates a directory artifacts if it does not exists.
          os.makedirs(os.path.dirname(self.dataIngestionConfig.train_data_path),exist_ok=True)

          df.to_csv(self.dataIngestionConfig.raw_data_path, index=False, header=True)
          train,test = train_test_split(df, test_size=0.3, random_state=41)
          train.to_csv(self.dataIngestionConfig.train_data_path, index=False, header=True)
          test.to_csv(self.dataIngestionConfig.test_data_path, index=False, header=True)

          logging.info('Ingestion of data is completed')

          return (
              self.dataIngestionConfig.train_data_path,
              self.dataIngestionConfig.test_data_path
          )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
        
