import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
model = load_model('bankchurnprediction.keras')

def pickler_reader(pickler_name ):
    with open(pickler_name ,'rb') as file:
        pickler = pickle.load(file)
    return pickler

cov_bool = {False:0 , True:1}
st.title("Bank Churn Prediction")
creditScore = st.slider(' Select Credit Score' , min_value = 300 ,max_value =900 , step=1)
geography = st.radio(' Select Country ' ,('France', 'Germany' ,'Spain'))
gender = st.radio(' Select Gender ' , ('Male' ,'Female'))
age = st.slider(' Select Age ' , min_value = 20 ,max_value =90 , step=1)
tenure = st.slider(' Select Tenure ' , min_value = 0 ,max_value =15 , step=1)
balance = st.slider(' Select Balance ' , min_value = 0 ,max_value =999999 , step=1)
products = st.slider(' Select No. of Products ' , min_value = 0 ,max_value =10 , step=1)
card = st.checkbox('Has Credit Card ? ' ,value=False)
memeber = st.checkbox('Is Active Member ? ' ,value=False)
salary = st.slider(' Select Estimated Salary ' , min_value = 0 ,max_value =999999 , step=1)

analys =  st.button('Check Eligibility .')

if analys:
    user_input = {
        'CreditScore': creditScore,
        'Geography' : geography , 
        'Gender' : gender ,
        'Age' : age ,
        'Tenure' : tenure ,
        'Balance' : balance,
        'NumOfProducts' : products , 
        'HasCrCard' : cov_bool.get(card) , 
        'IsActiveMember' : cov_bool.get(memeber),
        'EstimatedSalary' : salary
    }
    lable_encode = pickler_reader('Genderencoder.pkl')
    user_input['Gender'] = lable_encode.transform([user_input['Gender']])
    user_input = pd.DataFrame(user_input)
    one_hot_encode = pickler_reader('Geographyencoder.pkl')
    column_encode = one_hot_encode.transform(user_input[['Geography']]).toarray()
    column_encode_df = pd.DataFrame(column_encode , columns=one_hot_encode.get_feature_names_out(['Geography']))
    user_input = pd.concat([user_input.drop('Geography',axis=1 ) ,column_encode_df ],axis=1)
    scaller = pickler_reader('scallerencoder .pkl')
    scalled_data = scaller.transform(user_input)
    pridict = model.predict(scalled_data)  
    if pridict[0][0]*100 > 50:
        st.info(f'Customer Will Churn. Chances : {pridict[0][0] * 100:.2f} %'   )
    else:
        st.info(f"Customer will not churn. Chances: {pridict[0][0] * 100:.2f} %")