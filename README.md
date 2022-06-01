# StarBucks Capstone Project
## **Motivation**
This project utilize data set containing simulated data that mimics how people make purchasing decisions and how those decisions are influenced by promotional offers.
The purpose of this project is to identify whether a new user who receives an offer will use that offer or not. 
## **Instructions** :
If you see error when reading transcript.json you will need to run the command conda update pandas before reading in the files. 
This is because the version of pandas in your workspace cannot read in the transcript.json file correctly, but the newest version of pandas can. 
You can access the termnal and type `conda update pandas`, after finish, you should be able to read all datasets without any errors.
## **Libraries**
This project use built-in libraries for data cleaning ,text processing and data modeling:
`import pandas as pd`

`import numpy as np`

`import math`

`import json`

`import matplotlib.pyplot as plt`

`import seaborn as sns`

`% matplotlib inline`

`from datetime import datetime`

`from sklearn import preprocessing`

`from sklearn.pipeline import Pipeline,FeatureUnion`

`from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score`

`from sklearn.model_selection import train_test_split`

`from sklearn.model_selection import GridSearchCV`

`from sklearn.linear_model import LogisticRegression`

`from sklearn.svm import LinearSVC`

`from sklearn.tree import DecisionTreeClassifier`

`from sklearn.naive_bayes import GaussianNB`

`from sklearn.neighbors import KNeighborsClassifier`

`from sklearn.ensemble import RandomForestClassifier`
## **Table of content**
1. data

There are 3 datasets:

- portfolio.json # containing offer ids and meta data about each offer (duration, type, etc.)

- profile.json # demographic data for each customer

- transcript.json # records for transactions, offers received, offers viewed, and offers completed

2. notebook
- Starbucks_Capstone_notebook.ipynb # main code where all steps are processed

## **Conclusion**

In my analysis, I have cleaned 3 dataset profile, portfolio and transaction to get a combination dataset that has full information and can be used to build classification model.

I have chose random forest as the classifier for my model, since it has the best performance among all types of models that can be used for binary classification while comparing evaluation metrics like accuracy, precision, and recall.

In order to prevent data leakage and optimize hyperparameters of the model, I have used machine learning pipeline and run grid search and found out the best parameters are clf__n_estimators = 200 and scaler__with_mean = True. The final model has accuracy of 0.921.

I have written a blog post that communicates the critical steps that I have done and shares my key findings and processes.
You can find it here: https://seattlehouseprice.blogspot.com/2022/06/customer-behavior-prediction-using.html
## **Improvement**

- Time variable is not carefully considered in this project, there might be a deep dive into this feature to improve model performance or find other insights.
- Informational offer type is also a variable that has not been analyzed much since there is no reward or transaction amount related to this type. We might need more data, such as transaction amount before and after the offer was received.
