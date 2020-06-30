from pycaret.classification import load_model, predict_model
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


model = load_model('xgboost_for_deployment')
data1 = pd.read_csv('churn.csv')


def run():
    from PIL import Image
    image = Image.open('logo.png')
    image_digital_trust = Image.open('digital_trust.jpg')
    st.image(image , use_column_width = False)
    st.title('Telecom Churn Prediction App')

    file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

    if file_upload is not None:
        data = pd.read_csv(file_upload)
	data_new = pd.merge(data , data1 , how = 'inner' , on = 'phone number')
        predictions = predict_model(estimator=model,data=data_new)
        predictions.Label[predictions.Label  ==  0 ] = 'Non churn'
        predictions.Label[predictions.Label  ==  1 ] = 'Churn'
        g = sns.countplot(x = 'Label' , data = predictions )
        plt.title('Count of Churners and Non churners')
        for p in g.patches:
            g.annotate(format(p.get_height() ), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    st.pyplot()

if __name__ == '__main__':
    run()	
