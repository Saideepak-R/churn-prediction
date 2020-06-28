from pycaret.classification import load_model, predict_model
import streamlit as st
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pandas as pd


model = load_model('xgboost_for_deployment')


def run():
    from PIL import Image
    image = Image.open('logo.png')
    image_digital_trust = Image.open('digital_trust.jpg')
    st.image(image , use_column_width = False)
    st.title('Telecom Churn Prediction App')

    file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

    if file_upload is not None:
        data = pd.read_csv(file_upload)
        predictions = predict_model(estimator=model,data=data)
        st.write(predictions)

if __name__ == '__main__':
    run()
