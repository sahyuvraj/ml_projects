import streamlit as st  
import numpy as np  
import pickle
import pandas as pd  
y_predict = pickle.load(open('knn1.pkl','rb'))

st.title("KNN PREDICTION")
st.header("Purchased home or not")
st.subheader("fill the details below")
age = st.number_input('Enter your Age')
salary = st.number_input('Enter your salary')

if st.button('Predict'):
    query = pd.DataFrame([[age,salary]])
    knn = y_predict['model']
    prediction = knn.predict(query)
    if prediction == 0:
        st.write("NO YOU CAN'T PURCHASED")
    elif prediction == 1:
        st.write("YES YOU CAN PURCHASED")
    else:
        st.write("ERROR")
