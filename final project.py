# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec



# Load the dataset from the csv file using pandas
data = pd.read_csv("C:/Users/Icon/Desktop/ML2/creditcard.csv")

# Few rows of the data
data.head()

#checking missing values in data
data.isnull().sum()

# Print the shape of the data and describtion of data
print(data.shape)
data.describe()

#correlation matrix
cor_matrix=data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(cor_matrix, vmax = 1, square = True)
plt.title("Heat map")
plt.show()

#EDA
data['Class'].value_counts().plot(kind='pie')

#Non-Null Count and Data type 
data.info()

#Distribution plot of classes in responce variable and 'V17' (highest negative correlation to responce variable)
sns.distplot(data[data['Class']==0]['V17'], hist=False)
sns.distplot(data[data['Class']==1]['V17'], hist=False)

#Distribution plot of classes in responce variable and 'V11' (highest positive correlation to responce variable)
sns.distplot(data[data['Class']==0]['V11'], hist=False)
sns.distplot(data[data['Class']==1]['V11'], hist=False)

x=data.drop('Class', axis=1) #features
y=data['Class']              #target


import pandas as pd

# Pearson correlation coefficient
pearson_corr = x.corrwith(y, method='pearson')

print("Pearson Correlation Coefficients:\n", pearson_corr)

selected_features_drop = pearson_corr[(pearson_corr >= -0.015) & (pearson_corr <= 0.015)]
selected_features_drop

# Assuming 'x' is your main feature DataFrame and you want to drop columns 'V13', 'V15', 'V22', 'V23', 'V24', 'V25','V26' and 'Amount'
columns_to_drop = ['Time','V13','V15','V22','V23','V24','V25','V26','V28','Amount']
x = x.drop(columns=columns_to_drop)
x

new_data = pd.concat([x, data['Class']], axis=1)
new_data

# Determine number of fraud cases in dataset
fraud = new_data[new_data.Class == 1]   # Gives thoese rows which contain 1
valid = new_data[new_data.Class == 0]   # Gives thoese rows which contain 0

#shape of valid and fraud transaction data
print(f'shape of fraud data set is {fraud.shape}')
print(f'shape of valid data set is {valid.shape}')

outlierFraction = len(fraud)/float(len(data))
print(f'fraction of fraud out to total numbers of transaction is {outlierFraction}')
print(f'Number of Fraud cases= {len(fraud)}')
print(f'Number of Valid Transactions= {len(valid)}')

#Building sample from valid transaction data of same size as that of fraud transaction
valid_sample= valid.sample(n=492)
print(valid_sample.shape)
print(fraud.shape)

#concatenate above two data frame
new_data=pd.concat([valid_sample, fraud], axis=0) # row wise cocatenation
new_data_shuffled = new_data.sample(frac=1.0, random_state=42)
new_data_shuffled

#spliting data into feature and target
X=new_data_shuffled.drop('Class', axis=1) #features
Y=new_data_shuffled['Class']  #target

#spliting data into training and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2,stratify=Y,random_state=42)

#Traning Logestic regression model (Logistic regression model use for binary classification (0,1))
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

#Training logistic model with training data
model.fit(x_train, y_train)
#Accuracy score for training data
from sklearn.metrics import accuracy_score
y_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train_prediction, y_train)
print(f'Accuracy score for training data is {training_data_accuracy}')

#Accuracy score for test data
y_test_prediction = model.predict(x_test)

y_scores = model.predict_proba(x_test)[:,1]


test_data_accuracy = accuracy_score(y_test_prediction, y_test)
print(f'Accuracy score for test data is {test_data_accuracy}')
#print(f'Probabilities are {y_scores}')

#Confussion matrix
from sklearn.metrics import confusion_matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_test_prediction)
plt.figure(figsize =(8, 6))
sns.heatmap(conf_matrix, xticklabels = LABELS, 
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

#ROC Curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_scores)
fpr

import plotly.graph_objects as go
import numpy as np


# Generate a trace for ROC curve
trace0 = go.Scatter(
    x=fpr,
    y=tpr,
    mode='lines',
    name='ROC curve'
)

# Only label every nth point to avoid cluttering
n = 10  
indices = np.arange(len(thresholds)) % n == 0  # Choose indices where index mod n is 0

trace1 = go.Scatter(
    x=fpr[indices], 
    y=tpr[indices], 
    mode='markers+text', 
    name='Threshold points', 
    text=[f"Thr={thr:.2f}" for thr in thresholds[indices]], 
    textposition='top center'
)


# Diagonal line
trace2 = go.Scatter(
    x=[0, 1], 
    y=[0, 1], 
    mode='lines', 
    name='Random (Area = 0.5)', 
    line=dict(dash='dash')
)

data = [trace0, trace1, trace2]

# Define layout with square aspect ratio
layout = go.Layout(
    title='Receiver Operating Characteristic',
    xaxis=dict(title='False Positive Rate'),
    yaxis=dict(title='True Positive Rate'),
    autosize=False,
    width=600,
    height=600,
    showlegend=False
)

# Define figure and add data
fig = go.Figure(data=data, layout=layout)

# Show figure
fig.show()

#Optimum threshold value 
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Optimal threshold is:", optimal_threshold)

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_scores)
print(f'ROC_AUC Score is {roc_auc}')

#Pression, recall and F1 score
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score

prec = precision_score(y_test, y_test_prediction)

rec = recall_score(y_test, y_test_prediction)

f1 = f1_score(y_test, y_test_prediction)

print(f"The precision score is {prec}")

print(f"The recall score is {rec}")

print(f"The f1 score is {f1}")

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier()
classifier.fit(x_train, y_train)

predicted=classifier.predict(x_test)
print("\n Predicted value by decision tree:\n",predicted)

from sklearn.metrics import precision_score,recall_score,f1_score
DecisionTree= accuracy_score(y_test, predicted)
print("\n The Accuracy Score Using Algorithm Decision Tree Classifier without cross validation is : ", DecisionTree)

print(f'Precision_score by decision tree is {precision_score(y_test,predicted)}')
print(f'Recall_score by decision tree is {recall_score(y_test,predicted)}')
print(f'F1_score by decision tree is {f1_score(y_test,predicted)}')

param_dist = {"criterion": ["ginni","entropy"], "max_depth" : [1,2,3,4,5,6,7,None]}

from sklearn.model_selection import GridSearchCV
#K-Fold cross validation
grid = GridSearchCV(classifier, param_grid=param_dist, cv=10, n_jobs=-1)

grid.fit(x_train, y_train)

print(grid.best_estimator_)

print(grid.best_score_)

print(grid.best_params_)

#Random forest classifier

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)

import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

# Assuming you have your Random Forest model stored in the variable 'rf_model'
# and your test data (X_test) and true labels (y_test) available

# Predict probabilities on the test set
y_pred_prob = rf.predict_proba(x_test)[:, 1]

# Compute ROC curve
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_prob)

# Calculate AUC
roc_auc = auc(fpr_rf, tpr_rf)

# Generate a trace for ROC curve
trace0 = go.Scatter(
    x=fpr_rf,
    y=tpr_rf,
    mode='lines',
    name=f'ROC curve (AUC = {roc_auc:.2f})'
)

# Only label every nth point to avoid cluttering
n = 10
indices = np.arange(len(thresholds)) % n == 0  # Choose indices where index mod n is 0

trace1 = go.Scatter(
    x=fpr_rf[indices],
    y=tpr_rf[indices],
    mode='markers+text',
    name='Threshold points',
    text=[f"Thr={thr:.2f}" for thr in thresholds[indices]],
    textposition='top center'
)

# Diagonal line
trace2 = go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Random (Area = 0.5)',
    line=dict(dash='dash')
)

data = [trace0, trace1, trace2]

# Define layout with square aspect ratio
layout = go.Layout(
    title='Receiver Operating Characteristic',
    xaxis=dict(title='False Positive Rate'),
    yaxis=dict(title='True Positive Rate'),
    autosize=False,
    width=600,
    height=600,
    showlegend=False
)

# Define figure and add data
fig = go.Figure(data=data, layout=layout)

# Show figure
fig.show()

#Optimum threshold value 
optimal_idx_rf = np.argmax(tpr_rf - fpr_rf)
optimal_threshold_rf = thresholds[optimal_idx]
print("Optimal threshold is:", optimal_threshold_rf)

from sklearn.metrics import roc_auc_score
roc_auc_rf = roc_auc_score(y_test, y_scores)
print(f'ROC_AUC Score is {roc_auc_rf}')

y_pred_rf = rf.predict(x_test)

print(f'Accuracy of Random Forest Classifier on data is {accuracy_score(y_test,y_pred_rf)}')
print(f'Precision of Random Forest Classifier on data is {precision_score(y_test,y_pred_rf)}')
print(f'Recall score of Random Forest Classifier on data is {recall_score(y_test,y_pred_rf)}')
print(f'F1 score of Random Forest Classifier on data is {f1_score(y_test,y_pred_rf)}')

final_data = pd.DataFrame({'Models':['LR','DT','RF'],
              "ACC":[accuracy_score(y_test,y_test_prediction)*100,
                     accuracy_score(y_test,predicted)*100,
                     accuracy_score(y_test,y_pred_rf)*100
                    ]})

print(final_data)

plt.figure(figsize=(7, 4))
sns.barplot(x='Models', y='ACC', data=final_data)
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Different Models')
plt.show()

# Number of trees in random forest
n_estimators = [20,60,100,120]

# Number of features to consider at every split
max_features = [0.2,0.6,1.0]

# Maximum number of levels in tree
max_depth = [2,8,None]

# Number of samples
max_samples = [0.5,0.75,1.0]

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
              'max_samples':max_samples
             }
print(param_grid)

from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV

rf_grid = GridSearchCV(estimator = rf, 
                       param_grid = param_grid, 
                       cv = 5, 
                       verbose=2, 
                       n_jobs = -1)

rf_grid.fit(x_train,y_train)

print(rf_grid.best_params_)
print(rf_grid.best_score_)




