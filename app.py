import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


# import all the models

model = tf.keras.models.load_model('model.h5')

with open("Label_encoder_gender", 'rb') as file:
    Label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo", 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler", 'rb') as file:
    scaler = pickle.load(file)

# title
st.title("Customer Churn Prediction")

# input

Geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
CreditScore = st.number_input("CreditScore")
Gender = st.selectbox("Gender", Label_encoder_gender.classes_)
Age = st.slider("Age", 18, 99)
Tenure = st.slider("Tenure", 2, 10)
Balance = st.number_input("Balance")
NumOfProducts = st.slider("NumOfProducts", 1, 4)
HasCrCard = st.selectbox("HasCrCard", [0, 1])
IsActiveMember = st.selectbox("IsActiveMember", [0, 1])
EstimatedSalary = st.number_input("EstimatedSalary")

input_data = pd.DataFrame({
    "CreditScore": [CreditScore],
    "Gender": [Label_encoder_gender.transform([Gender])[0]],
    "Age": [Age],
    "Tenure": [Tenure],
    "Balance": [Balance],
    "NumOfProducts": [NumOfProducts],
    "HasCrCard": [HasCrCard],
    "IsActiveMember": [IsActiveMember],
    "EstimatedSalary": [EstimatedSalary]
})

geo_encoded_df = onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded_df, columns=onehot_encoder_geo.get_feature_names_out())

input_df = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df], axis=1)

scaled_input = scaler.transform(input_df)

prediction = model.predict(scaled_input)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write("this person churn")
else:
    st.write("this is not churning")
