import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
@st.cache()
def prediction(model,ri,na,mg,al,si,k,ca,ba,fe):
  glass_type = model.predict([ri,na,mg,al,si,k,ca,ba,fe])
  glass_type = glass_type[0]
  if glass_type == 1:
    return 'Building windows float processed'.upper()
  elif glass_type ==2:
    return 'Building windows non float processed'.upper()
  elif glass_type == 3:
    return 'vehicle windows float windows'.upper()
  elif glass_type == 4:
    return 'vehicle windows non float windows'.upper()
  elif glass_type == 5:
    return 'container'.upper()
  elif glass_type ==6:
    return 'tableware'.upper()
  else:
    return 'headlamp'.upper()
st.title('Glass Type Predictor')
st.sidebar.title('Exploratory Data Analysis')
if st.sidebar.checkbox('Show Raw Data'):
  st.subheader('Full Data Set')
  st.dataframe(glass_df)
st.subheader('Scatter Plot')
features_list = st.sidebar.multiselect('Select the x-axis values',
                                       ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
st.set_option('deprecation.showPyplotGlobalUse', False)
for i in features_list:
  st.subheader(f'Scatter Plot between: {i} glass type')
  plt.figure(figsize = (12,6))
  sns.scatterplot(x=i,y='GlassType',data=glass_df)
  st.pyplot()
st.sidebar.subheader('Histogram')
hist_features = st.sidebar.multiselect('Select features to select Histogram',
                                       ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
for i in hist_features:
  st.subheader(f'Histogram for {i}')
  plt.figure(figsize = (12,6))
  plt.hist(glass_df[i],bins = 'sturges',edgecolor='black')
  st.pyplot()
st.sidebar.subheader('Box Plot')
box_features = st.sidebar.multiselect('Select features to select Box Plot',
                                       ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
for i in box_features:
  st.subheader(f'Box Plot for {i}')
  plt.figure(figsize = (12,6))
  sns.boxplot(glass_df[i])
  st.pyplot()