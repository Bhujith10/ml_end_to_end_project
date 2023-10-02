import pandas as pd
import streamlit as st
from src.logger import logging
from src.pipeline.predict_pipeline import PredictPipeline,CustomData

st.title('Student Performance prediction')
gender = st.selectbox('Gender',('male','female'))
race_ethnicity = st.selectbox('Race/Ethnicity',('group A', 'group B', 'group C', 'group D', 'group E'))
parental_education = st.selectbox('Parental Level of Education',
                                  ("bachelor's degree", 'some college', "master's degree",
                                   "associate's degree", 'high school', 'some high school'))
lunch = st.selectbox('Lunch',('standard', 'free/reduced'))
test_preparation_course = st.selectbox('Test Preparation course',('none', 'completed'))
reading_score = st.number_input('Enter the reading score in the range of 0-100')
writing_score = st.number_input('Enter the writing score in the range of 0-100')

data = CustomData(gender=gender,
                  race_ethnicity=race_ethnicity,
                  parental_education=parental_education,
                  lunch=lunch,
                  test_preparation_course=test_preparation_course,
                  reading_score=reading_score,
                  writing_score=writing_score)
pred_dataframe = data.create_dataframe()
predict_pipeline = PredictPipeline()
result = predict_pipeline.predict(pred_dataframe)

st.subheader('Your Math Score based on the model is')

st.write(str(round(result[0],2)))

