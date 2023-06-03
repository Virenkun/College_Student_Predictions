import streamlit as st
import pandas as pd
import numpy as np
import joblib
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder



# Load the trained model and transformers
rc = joblib.load('model.pkl')
trf1 = joblib.load('transformer.pkl')
pca = joblib.load('pca.pkl')
lb = joblib.load('label_encoder.pkl')

column_names = (['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
       'SectionID', 'Topic', 'Semester', 'Relation', 'raisedhands',
       'VisITedResources', 'AnnouncementsView', 'Discussion',
       'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
       'StudentAbsenceDays'])

# Create a dictionary to map class labels to user-friendly names
class_labels = {
    0: 'L',
    1: 'M',
    2: 'H'
}







# Create a function to preprocess user inputs
def preprocess_input(inputs):
    df = pd.DataFrame([inputs], columns=column_names)
    transformed_inputs = trf1.transform(df)
    transformed_inputs = pca.transform(transformed_inputs)
    return transformed_inputs


# Create a function to make predictions
def predict_category(inputs):
    transformed_inputs = preprocess_input(inputs)
    prediction = rc.predict(transformed_inputs)
    return class_labels[prediction[0]]


# Create the Streamlit app
def main():
    st.title("College Student Prediction System")

    # Create input fields for user inputs
    inputs = {}
    option = st.selectbox(
        'Your Gender',
        ('M', 'F'))
    inputs['gender'] = option

    option2 = st.selectbox(
        'Your Nationality',
        ('KW',
 'lebanon',
 'Egypt',
 'SaudiArabia',
 'USA',
 'Jordan',
 'venzuela',
 'Iran',
 'Tunis',
 'Morocco',
 'Syria','Palestine',
 'Iraq',
 'Lybia'))

    inputs['NationalITy'] = option2

    option3 = st.selectbox(
        'PlaceofBirth?',
        ('KuwaIT',
 'lebanon',
 'Egypt',
 'SaudiArabia',
 'USA',
 'Jordan',
 'venzuela',
 'Iran',
 'Tunis',
 'Morocco',
 'Syria','Iraq',
 'Palestine',
 'Lybia'))

    inputs['PlaceofBirth'] = option3

    option4 = st.selectbox(
        'StageID ?',
        ('lowerlevel', 'MiddleSchool', 'HighSchool'))

    inputs['StageID'] = option4

    option5 = st.selectbox(
        'GradeID',
        ('G-04',
 'G-07',
 'G-08',
 'G-06',
 'G-05',
 'G-09',
 'G-12',
 'G-11',
 'G-10',
 'G-02'))

    inputs['GradeID'] = option5

    option6 = st.selectbox(
        'SectionID?',
        ('A', 'B', 'C'))

    inputs['SectionID'] = option6

    option7 = st.selectbox(
        'Topic?',
        ('IT',
 'Math',
 'Arabic',
 'Science',
 'English',
 'Quran',
 'Spanish',
 'French',
 'History',
 'Biology',
 'Chemistry','Geology'))

    inputs['Topic'] = option7

    option8 = st.selectbox(
        'Semester?',
        ('F', 'S'))

    inputs['Semester'] = option8

    option9 = st.selectbox(
        'Relation?',
        ('Father', 'Mum'))

    inputs['Relation'] = option9

    option10 = st.number_input('Insert Number of Raised Hands', step=1)

    inputs['raisedhands'] = option10

    option11 = st.number_input('Insert Number of VisITedResources', step=1)

    inputs['VisITedResources'] = option11

    option12 = st.number_input('Insert AnnouncementsView', step=1)

    inputs['AnnouncementsView'] = option12

    option13 = st.number_input('Insert a number of Discussion', step=1)

    inputs['Discussion'] = option13

    option14 = st.selectbox(
        'ParentAnsweringSurvey?',
        ('Yes', 'No'))

    inputs['ParentAnsweringSurvey'] = option14

    option15 = st.selectbox(
        'ParentschoolSatisfaction?',
        ('Good', 'Bad'))

    inputs['ParentschoolSatisfaction'] = option15

    option16 = st.selectbox(
        'StudentAbsenceDays?',
        ('Under-7', 'Above-7'))

    inputs['StudentAbsenceDays'] = option16

    # Create a button to trigger the prediction
    if st.button("Predict"):
        prediction = predict_category(inputs)
        st.success(f"The predicted category is: {prediction}")


if __name__ == '__main__':
    main()