import streamlit as st # type: ignore
import joblib as jb # type: ignore

vectorizer = jb.load('vectorizer.jb')
model = jb.load('LR_model.jb')

st.title('Fake News Detection APP')
st.write('Please enter an news you need to verify if it is fake or real')

string = st.text_area('Enter the news here:','')

if st.button('Check News'):
    if string.strip():
        string = vectorizer.transform([string])
        result = model.predict(string)
        if result[0] == 1:
            st.success('WOW !! it''s Real News')
        else:
            st.error('Ohh no!!! It''s Fake News')
    else:
        st.warning('Please enter someting to predict the news')
    