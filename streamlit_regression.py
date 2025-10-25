import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle


model = tf.keras.models.load_model('regression_model.h5')

# load the encoder and scaler
with open('ohe_geo.pkl', 'rb') as file:
    ohe_encoder = pickle.load(file)
    
with open('scaler_salary.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
with open('label_encoder2.pkl', 'rb') as file:
    label_encoder = pickle.load(file)        
    
#streamlit app
st.title('Estimated Salary Prediction')

# user input
geography = st.selectbox('Geography', ohe_encoder.categories_[0]) 
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Is Exited ?', [0, 1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# prepare the input data

input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'Exited': exited
}

# encode the Geography feature

geo_encoded = ohe_encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns = ohe_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([pd.DataFrame([input_data]), geo_encoded_df], axis = 1)
# scaling the input data
input_scaled = scaler.transform(input_data)

# predict Churn data
prediction  = model.predict(input_scaled)
predicted_salary = prediction[0][0]

st.write(f'Estimated Salary Prediction: {predicted_salary:.2f}')