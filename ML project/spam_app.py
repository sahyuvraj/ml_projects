import streamlit as st    
import pickle 
import pandas as pd 
import numpy as np
V= pickle.load(open("spam_count_vector.pkl",'rb'))
y_predict = pickle.load(open('spam.pkl','rb'))

st.header("Email/SMS Spam Classifier")
message = st.text_input("Enter the message")
if st.button('Predict'):
    data = V.transform([message])
    result = y_predict.predict(data)
    if (result == 0):
        st.write("THIS Email/SMS IS NOT SPAM")
    else:
        st.write("THIS Email/SMS IS A SPAM")



