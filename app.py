from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = load_model('xgboost_for_deployment')
image = Image.open('logo.png')
image_digital_trust = Image.open('digital_trust.jpg')

def run():
	from PIL import Image
	st.image(image , use_column_width = False)
#st.sidebar.image(image  , use_column_width = False)
    
#st.sidebar.info('ROC insights')
#st.sidebar.success('https://www.subex.com/act/insights/')
#st.sidebar.image(image_digital_trust)
	st.title('Telecom Churn Prediction App')

	file_upload = st.file_uploader('Upload csv file for predictions', type = ['csv'])
    
	if file_upload is not None:
		data = pd.read_csv(file_upload)
		predictions = predict_model(estimator = model , data = data)
	#predictions1 = predictions['Label'].map(lambda x : 'Not churn' if x == 0 else 'Churn' )
	#st.write(predictions)
		predictions.Label[predictions.Label  ==  0 ] = 'Non churn'
		predictions.Label[predictions.Label  ==  1 ] = 'Churn'
		g = sns.countplot(x = 'Label' , data = predictions )
		plt.title('Count of Churners and Non churners')
		for p in g.patches:
			g.annotate(format(p.get_height() ), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
	st.pyplot()

if __name__ == '__main__':
    run() 