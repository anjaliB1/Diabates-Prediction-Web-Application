import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('./trained_mode.sav', 'rb'))

def diabetes_prediction(input_data):
    input_data_np = np.asarray(input_data)
    input_data_reshaped = input_data_np.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'

def main():
    st.title('Diabetes Prediction Web Application')
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level (mg/dL)')
    BloodPressure = st.text_input('Blood Pressure (mm Hg)')
    SkinThickness = st.text_input('Skin Thickness (mm)')
    Insulin = st.text_input('Insulin Level (μu/ml)')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age (years)')
    
    diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])   
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()