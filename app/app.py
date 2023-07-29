# Import dependencies
import streamlit as st
import pandas as pd
import os
import pickle
import json
from utils import transform_data
from matplotlib import pyplot as plt
import seaborn as sns


# Set title and write output
st.title('Predict Churn 🚀')
st.write('Hit Predict to determine if your customer is likely to churn')

# Load Schema
with open('schema.json', 'r') as f:
    schema = json.load(f)
#st.write(schema)

#Set up column orders
column_order_in = list(schema['column_info'].keys())[:-1]
column_order_out = list(schema['transformed_columns']['transformed_columns'])
#st.write(column_order_out)

# create a sidebar section
st.sidebar.info('Update these features to predict based on your customer!')

# Collect input features
options = {}

for column, column_propierties in schema['column_info'].items():
    if column == 'churn':
        pass
    
    # Create numerical sliders
    elif column_propierties['dtype'] == 'int64' or column_propierties['dtype']== 'float64':
        min_val, max_val = column_propierties['values']
        data_type = column_propierties['dtype']

        feature_mean = (min_val+max_val) / 2
        if data_type == 'int64':
            feature_mean = int(feature_mean)
        
        options[column] = st.sidebar.slider(column, min_val, max_val, value= feature_mean)

        # Create categorical select boxes
    elif column_propierties['dtype'] == 'object':
        options[column] = st.sidebar.selectbox(column, column_propierties['values'])

# Load model and encoder
model_path = os.path.join('..', 'models', 'experiment_1', 'xg.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

encoder_path = os.path.join('..', 'models', 'experiment_1', 'encoder.pkl')
with open(encoder_path, 'rb') as f:
    onehot = pickle.load(f)

# Mean evening minutes value
mean_eve_mins = 200.29

#st.write(options)

# Create a predict button and Make a prediction
if st.button('Predict'):
    
    #Convert options to df
    scoring_data = pd.Series(options).to_frame().T
    scoring_data = scoring_data[column_order_in]
    

    # check datatypes
    for column, column_propierties in schema['column_info'].items():
        if column != 'churn':
            dtype = column_propierties['dtype']
            scoring_data[column] = scoring_data[column].astype(dtype)
   

    # Apply data transformations
    scoring_sample = transform_data(scoring_data, column_order_out, mean_eve_mins, onehot)
    #st.write(scoring_sample)

    # Render predictions
    prediction = model.predict(scoring_sample)
    st.write('Prediction Outcome')
    st.write(prediction)
    st.write('Client Details')
    st.write(options)

# Save Historical Data
try:
    historical = pd.Series(options).to_frame().T
    historical['prediction'] = prediction
    if os.path.isfile('historical_data.csv'):
        historical.to_csv('historical_data.csv', mode='a', header= False, index= False)
    else:
        historical.to_csv('historical_data.csv', header= True, index= False)
except Exception as e:
    pass

# Show historical predictions on dataframe on the app
st.header('Historical Outcomes')
if os.path.isfile('historical_data.csv'):
    hist = pd.read_csv('historical_data.csv')
    st.dataframe(hist)
    fig, ax = plt.subplots()
    sns.countplot(x='prediction', data= hist, ax=ax).set_title('Historical Predictions')
    st.pyplot(fig)
else:
    st.write('No historical data')

