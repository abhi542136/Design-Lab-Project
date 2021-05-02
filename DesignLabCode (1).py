
# coding: utf-8

# # Simulation of the Design Lab Project

# # Importing the Different Libraries

# In[2]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import pandas as pd
import plotly as plot
import plotly.express as px
import plotly.graph_objs as go

#import cufflinks as cf
import matplotlib.pyplot as plt
#import seaborn as sns
import os
from sklearn.metrics import accuracy_score
import plotly.offline as pyo
from plotly.offline import init_notebook_mode, plot, iplot


# # Acess the data of different people and storing it in 'heart' data frame

# In[3]:

heart = pd.read_csv(r'C:\Users\abhishek\DesignLab\Project_Design_Lab\heart.csv')


# # Let's see how our data look like in Tabular form

# In[4]:

heart


# In[5]:

info = ["age", "1: male, 0: female", "chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic", "resting blood pressure", " serum cholestoral in mg/dl", "fasting blood sugar > 120 mg/dl",
        "resting electrocardiographic results (values 0,1,2)", " maximum heart rate achieved", "exercise induced angina", "oldpeak = ST depression induced by exercise relative to rest", "the slope of the peak exercise ST segment", "number of major vessels (0-3) colored by flourosopy", "thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]


# # Dataset Description

# In[6]:

for i in range(len(info)):
    print(heart.columns[i]+":\t\t\t"+info[i])


# In[7]:

heart['target']


# In[8]:

heart.groupby('target').size()


# In[9]:

heart.groupby('target').sum()


# In[10]:

heart.shape


# In[11]:

heart.size


# # Statistical Info of the data set

# In[12]:

heart.describe()


# In[13]:

heart.info()


# In[14]:

heart['target'].unique()


# In[15]:

heart.hist(figsize=(14, 14))
plt.show()


# In[17]:

numeric_columns = ['trestbps', 'chol', 'thalach', 'age', 'oldpeak']
heart['target']


# In[18]:

target_temp = heart.target.value_counts()

print(target_temp)


# In[19]:

fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.show()


# # Data Pre-Processing

# In[20]:

heart['target'].value_counts()


# In[21]:

heart['target'].isnull()


# In[22]:

heart['target'].sum()


# In[23]:

heart['target'].unique()


# In[24]:

heart.isnull().sum()


# # Storing the data in X and y

# In[25]:

X, y = heart.loc[:, :'thal'], heart.loc[:, 'target']


# In[26]:

X


# In[27]:

y


# In[28]:

X.shape


# In[29]:

y.shape


# In[30]:

X = heart.drop(['target'], axis=1)


# In[31]:

X


# # Splitting the data into train and test for training and testing

# In[77]:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=10, test_size=0.2, shuffle=True)


# In[78]:

X_test


# In[79]:

y_test


# In[80]:

print("train_set_x shape: " + str(X_train.shape))
print("train_set_y shape: " + str(y_train.shape))
print("test_set_x shape: " + str(X_test.shape))
print("test_set_y shape: " + str(y_test.shape))


# # MODEL

# # 1. Decision Tree Classifier

# In[81]:

Catagory = ['No,You do not have Heart Disease...',
            'Yes you have Heart Disease...']


# In[82]:

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# # Finding the accuracy in Decision Tree Model

# In[83]:

prediction = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, prediction)*100


# In[84]:

accuracy_dt


# # We got 75.40 % accuracy in Decision Tree model

# In[85]:

print("Accuracy on training set: {:.3f}".format(dt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(dt.score(X_test, y_test)))


# In[86]:

y_test


# In[87]:

prediction


# # Testing with new data to verify result

# In[88]:

X_DT = np.array([[35, 1, 0, 126, 282, 0, 0, 156, 1, 0, 2, 0, 3]])
X_DT_prediction = dt.predict(X_DT)


# In[89]:

X_DT_prediction[0]


# # Yeah, Get The Result of new feeded data Here !!!

# In[90]:

print(Catagory[int(X_DT_prediction[0])])


# # Feature Importance in Decision Trees

# In[91]:

print("Feature importances:\n{}".format(dt.feature_importances_))


# In[92]:

def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8, 6))
    n_features = 13
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)


plot_feature_importances_diabetes(dt)
plt.savefig('feature_importance')


# # 2. KNN

# # Training with KNN 

# In[111]:

sc = StandardScaler().fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[112]:

X_test_std


# In[113]:

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_std, y_train)


# # Calculating Accuracy in KNN Model

# In[102]:

prediction_knn = knn.predict(X_test_std)
accuracy_knn = accuracy_score(y_test, prediction_knn)*100


# In[103]:

accuracy_knn


# # We got 85.2 % Accurcy in KNN Model

# In[104]:

print("Accuracy on training set: {:.3f}".format(knn.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(knn.score(X_test, y_test)))


# # Let's See how Accuracy varies with the value of k in KNN Model

# In[105]:

k_range = range(1, 26)
scores = {}
scores_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std, y_train)
    prediction_knn = knn.predict(X_test_std)
    scores[k] = accuracy_score(y_test, prediction_knn)
    scores_list.append(accuracy_score(y_test, prediction_knn))
scores


# # In the above output we can see that the accuracy is maximum at k=5

# In[106]:

plt.plot(k_range, scores_list)


# # Input the heart data below to get the result using KNN Model

# In[107]:

X_knn = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
X_knn_std = sc.transform(X_knn)
X_knn_prediction = dt.predict(X_knn)


# In[108]:

X_knn_std


# In[109]:

(X_knn_prediction[0])


# # Result of the user's input using KNN Model

# In[110]:

print(Catagory[int(X_knn_prediction[0])])


# 
