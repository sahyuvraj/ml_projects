import streamlit as st 
import pickle
import numpy as np
import pandas as pd
pipe = pickle.load(open('lrmodel.pkl','rb'))
car = pickle.load(open('df.pkl','rb'))

st.title("CARS PRICE PREDICTOR")
st.header("PREDICT THE PRICE OF CAR YOU WANT TO SELL.")
st.subheader("TRY FILLING THE DETAILS BELOW")
company = st.selectbox('company',car['company'].unique())

model = st.selectbox('name',car['name'].unique())
yr_of_purchase = st.number_input('insert year of purchase')
fuel_type = st.selectbox('fuel_type',car['fuel_type'].unique())
kms_driven = st.number_input('insert kms driven')
if st.button('Predict Price'):
    query = pd.DataFrame([[model,company,fuel_type,yr_of_purchase,kms_driven]],columns=['name','company','fuel_type','year','kms_driven'])
    st.title(int(pipe.predict(query)))