### Importing Libraries

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression

import altair as alt

### Reading Dataset

dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')



### Data Preprocessing

dataset['SeniorCitizen'].replace([1, 0],['Yes', 'No'],inplace=True)


## TotalCharges has null values.. (in string * ' ' * )
null_TotalCharges = dataset[dataset['TotalCharges'] == ' ']
dataset.drop(index = null_TotalCharges.index, inplace = True)


## Changing TotalCharges Dtype from Object to Float
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'])


## customerID is not a feature
dataset.drop(columns={'customerID'}, inplace = True)
## Selecting categorical features
categorical_columns = dataset.select_dtypes(include='object').columns 


##### Streamlit #####


#########################



## Encoding categorical features using label encoder
le = LabelEncoder()
encoded_dataset = dataset.copy()
encoded_dataset[categorical_columns] = encoded_dataset[categorical_columns].apply(le.fit_transform)






### Feature Engineering and Train/Test Split

X = encoded_dataset.iloc[:,:-1]
y = encoded_dataset['Churn'].values
print(X.shape, y.shape)

ct = ColumnTransformer(transformers=[('scalar', StandardScaler(), ['tenure',	'MonthlyCharges',	'TotalCharges'])], remainder='passthrough')
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



### Model Construction

clf = LogisticRegression()
clf.fit(X_train, y_train)



### Model Evaluation

y_pred = clf.predict(X_test)

print('***Training Result***\n')
print(classification_report(y_train, clf.predict(X_train)))
confusion_matrix(y_train,clf.predict(X_train))

print('***Testing Result***\n')
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)





##### Streamlit #####

## Visualization


st.header('Make your own prediction', divider='violet')

empty_list = {} ## for storing custom data

## radio
cols = st.columns(4, gap='small') ## to put each radiobox in separate columns
idx = 0
for i in categorical_columns:
    if i == 'Churn':
         break
    with cols[idx]:
        with st.container(height= 150, border=True):  ## put into container first for better looking
            selection = dataset[i].unique()
            choice = st.radio(i, selection)
            empty_list[i] = choice
    idx = idx + 1
    if idx == 4:
        idx = 0
    
## slider    
for j in ['MonthlyCharges','TotalCharges','tenure']:
    choice = st.slider(j, float(dataset[j].min()), float(dataset[j].max()), float(dataset[j].mean()))
    empty_list[j] = choice



# # # # # # # # 



input_data = pd.DataFrame([empty_list])  ## make dict with customer data to dataframe


st.dataframe(input_data.T, use_container_width=True) ## show df

categorical_columns = categorical_columns.drop('Churn')



'''comment ... not working'''
## Label Encoding user choice data 
input_data[categorical_columns] = input_data[categorical_columns].apply(le.fit_transform)

'''comment ... also not working'''
## Scaling user choice data
input_data = ct.fit_transform(input_data)

st.subheader("Your data after Encoding and Scaling")
st.dataframe(input_data)





# # Make predictions
prediction = clf.predict(input_data)

# # Display the prediction
answer = pd.Series(prediction).map({0: 'NO', 1: 'Yes'})
st.markdown(f'The Prediction : {answer[0]}')
'''comment ... always showing No or 0'''
