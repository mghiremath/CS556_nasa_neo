#!/usr/bin/env python
# coding: utf-8

# In[5]:


# importing required libaries

# for handling dataframe operations
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA

# for machine learning model building
from sklearn import linear_model
from sklearn.svm import SVC


# metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')

# to get the status of jobs in loop
from tqdm import tqdm

# to filter-out all the warnings given by python modules
import warnings
warnings.filterwarnings("ignore")


# In[6]:


# reading the data from csv and looking first 5 rows
data = pd.read_csv("NASA.csv")
data.head(5)


# - by quickly observing we can find this things from data:
# 
#     - We can drop features id, name, orbiting_body, sentry_object. Since they are almost unique values, so it can't be useful for us to classify data points.
#     - Hazardous column consists values either true or false (string), we can convert them to 0 or 1.

# In[7]:


hazardous_encoder = preprocessing.LabelEncoder() # initializing label encoder object 
data['hazardous'] = hazardous_encoder.fit_transform(data.hazardous) # transforming false to 0 and true to 1 in hazardous column
data.drop(['id', 'name', 'orbiting_body', 'sentry_object'], axis=1, inplace=True) # dropping the columns consists of unique values
data.head()


# ### Checking for missing data

# In[8]:


data.isnull().sum(axis = 0)


#  - This indicating we don't have any missing data in our features

# ### Glimpse of feature stastics:

# In[9]:


data.describe()


# ### Exploratory Data Analysis:
#  - Here we will check the distributions of each column and we will remove potential outliers in our dataset

# In[10]:


# est_diameter_min:
plt.rcParams['figure.figsize'] = (10, 5)
sns.distplot(data['est_diameter_min'])
plt.show()


#  - This distribution of est_diameter is right skewed, so there is a high chance of potential outliers. 
#  - By using Z-score concept, we can drop values which are lesser than mu - 3*sigma and higher than mu + 3*sigma

# In[11]:


lower_bound = data['est_diameter_min'].quantile(0.01) 
upper_bound  = data['est_diameter_min'].quantile(0.99)

data = data[(data["est_diameter_min"] < upper_bound) & (data["est_diameter_min"] > lower_bound)]
data.shape


# In[12]:


# We are gonna apply same technique on every numeric column in our data:
remaining_columns = ["est_diameter_max", "relative_velocity", "miss_distance", "absolute_magnitude"]
for i in remaining_columns:
    lower_bound = data[i].quantile(0.01) 
    upper_bound  = data[i].quantile(0.99)
    data = data[(data[i] < upper_bound) & (data[i] > lower_bound)]


#  <b> Let's Analyze Target Feature "hazardous" </b>

# In[13]:


# check the count of each category in target feature:
data.hazardous.value_counts()


# In[14]:


print("Percentage of non hazardous data points: ", data.hazardous.value_counts()[0]/data.shape[0])
print("Percentage of hazardous data points: ", data.hazardous.value_counts()[1]/data.shape[0])


#  - This is clear evident of unbalanced data. 
#  - Why?
#   - In the given data 90% of the data points are belong to class 0 (non hazardous) and only 10% of data points are belongs to class 1 (hazardous)
#   - We can handle this situation by giving more weight to minority class, which is hazardous

# ### Splitting the data:

# In[15]:


# Here are we are using 80% of data for training and 20% of data as testing with random state = 4 (to reproduce the same results)
x, y = data.iloc[:, 0:-1], data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)


# ### Scaling the features:

# In[16]:


# we will check for both normalization and standardization
# In standardization data will be scaled between [-1, 1]

# standardization
x_train_stand, x_test_stand = x_train.values, x_test.values # returns a numpy array
standard_scaler = preprocessing.StandardScaler() # initializing the StandardScaler class object
x_train_stand = standard_scaler.fit_transform(x_train_stand)
x_test_stand = standard_scaler.transform(x_test_stand)

# Normalization
x_train_norm, x_test_norm = x_train.values, x_test.values # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler() # initializing the min max scaler class object
x_train_norm = min_max_scaler.fit_transform(x_train_norm)
x_test_norm = min_max_scaler.transform(x_test_norm)


# ### Bulding models:

# #### 1. Logistic Regrssion

# #### Hyperparameter tuning:

# In[30]:


# expermiment 1: Building Logistic Regression model
 # 1.1 using the features which are normalized
 # 1.2 using the fatures which are standardized

hyper_parameter = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 100, 1000, 10000]
train_auc_normalized_data, train_auc_stdized_data = [], []
for i in tqdm(hyper_parameter):
    clf = linear_model.SGDClassifier(loss="log", 
                                     penalty="l2", 
                                     alpha=i,
                                     n_jobs=-1,
                                     class_weight='balanced', random_state=4)
    clf.fit(x_train_stand, y_train)
    y_train_pred = clf.predict_proba(x_train_stand)[:,1]    
    train_auc_stdized_data.append(roc_auc_score(y_train, y_train_pred))
    
plt.semilogx(hyper_parameter, train_auc_stdized_data, label='AUC plot')
plt.scatter(hyper_parameter, train_auc_stdized_data, label='AUC')

plt.xlabel("alpha: hyperparameter")
plt.ylabel("AUC")
plt.title("Train AUC vs Hyperparameter")

plt.legend()
plt.show()
plt.grid='True'


#  - We are getting good results with Standardized data, with alpha = 10^-3

# In[17]:


# Building the model with the best alpha value:

clf = linear_model.SGDClassifier(loss="log", 
                                 penalty="l2", 
                                 alpha=10**-3,
                                 n_jobs=-1,
                                 class_weight='balanced', random_state=4)
clf = clf.fit(x_train_stand, y_train)
y_train_pred = clf.predict(x_train_stand) 
y_test_pred = clf.predict(x_test_stand)

print("Accuray for training data is: {0} and for test data is: {1}".format(accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)))
print("\n")
print("Precision for training data is: {0} and for test data is: {1}".format(precision_score(y_train, y_train_pred), precision_score(y_test, y_test_pred)))
print("\n")
print("Recall for training data is: {0} and for test data is: {1}".format(recall_score(y_train, y_train_pred), recall_score(y_test, y_test_pred)))
print("\n")
print("F1 Score for training data is: {0} and for test data is: {1}".format(f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)))
print("\n")
print("AUC Score for training data is: {0} and for test data is: {1}".format(roc_auc_score(y_train, y_train_pred), roc_auc_score(y_test, y_test_pred)))


# In[29]:


result = pd.DataFrame(data=np.array([y_test, y_test_pred]).T, columns=["original", "predicted"])
result.to_csv("logistic_regression_results.csv", index=False)


# ####  2. SVM

# In[35]:


model = SVC(class_weight='balanced')
param_grid = {'C':[1, 10, 100], 'kernel':['rbf', 'poly']}
grid = GridSearchCV(model, param_grid, cv=3, return_train_score=True, verbose=5)
grid.fit(x_train_norm,y_train)


# In[36]:


print("The best accuracy is: {0} and the best model parameters are: {1}".format(grid.best_score_, grid.best_params_))


# In[30]:


clf = SVC(C=100, kernel='rbf', class_weight='balanced')
clf = clf.fit(x_train_norm, y_train)
y_train_pred = clf.predict(x_train_norm) 
y_test_pred = clf.predict(x_test_norm)

print("Accuray for training data is: {0} and for test data is: {1}".format(accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)))
print("\n")
print("Precision for training data is: {0} and for test data is: {1}".format(precision_score(y_train, y_train_pred), precision_score(y_test, y_test_pred)))
print("\n")
print("Recall for training data is: {0} and for test data is: {1}".format(recall_score(y_train, y_train_pred), recall_score(y_test, y_test_pred)))
print("\n")
print("F1 Score for training data is: {0} and for test data is: {1}".format(f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)))
print("\n")
print("AUC Score for training data is: {0} and for test data is: {1}".format(roc_auc_score(y_train, y_train_pred), roc_auc_score(y_test, y_test_pred)))


# In[31]:


result = pd.DataFrame(data=np.array([y_test, y_test_pred]).T, columns=["original", "predicted"])
result.to_csv("svm_results_without_pca.csv", index=False)


# ### Dimensionality reduction using PCA

# In[32]:


pca = PCA(n_components=2) # initializing the pca class object with 2 components
pca.fit(x_train_stand)
x_train_dim_reduced = pca.transform(x_train_stand) # dimenssions reduced train data
x_test_dim_reduced = pca.transform(x_test_stand) # dimenssions reduced test data


# #### Let's build new model on dimenssionality reduced data:

# In[22]:


model = SVC(class_weight='balanced')
param_grid = {'C':[1, 10, 100], 'kernel':['rbf', 'poly']}
grid = GridSearchCV(model, param_grid, cv=3, return_train_score=True, verbose=5)
grid.fit(x_train_dim_reduced,y_train)


# In[23]:


print("The best accuracy is: {0} and the best model parameters are: {1}".format(grid.best_score_, grid.best_params_))


# In[33]:


clf = SVC(C=1, kernel='poly', class_weight='balanced')
clf.fit(x_train_dim_reduced, y_train)
y_train_pred = clf.predict(x_train_dim_reduced) 
y_test_pred = clf.predict(x_test_dim_reduced)


# ##### Decision boundary:

# In[19]:


h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = x_test_dim_reduced[:, 0].min() - 1, x_test_dim_reduced[:, 0].max() + 1
y_min, y_max = x_test_dim_reduced[:, 1].min() - 1, x_test_dim_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis('off')
plt.scatter(x_test_dim_reduced[:,0], x_test_dim_reduced[:, 1], c=y_test, cmap=plt.cm.Paired)


# In[24]:


print("Accuray for training data is: {0} and for test data is: {1}".format(accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)))
print("\n")
print("Precision for training data is: {0} and for test data is: {1}".format(precision_score(y_train, y_train_pred), precision_score(y_test, y_test_pred)))
print("\n")
print("Recall for training data is: {0} and for test data is: {1}".format(recall_score(y_train, y_train_pred), recall_score(y_test, y_test_pred)))
print("\n")
print("F1 Score for training data is: {0} and for test data is: {1}".format(f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)))
print("\n")
print("AUC Score for training data is: {0} and for test data is: {1}".format(roc_auc_score(y_train, y_train_pred), roc_auc_score(y_test, y_test_pred)))


# In[34]:


result = pd.DataFrame(data=np.array([y_test, y_test_pred]).T, columns=["original", "predicted"])
result.to_csv("svm_results_with_pca.csv", index=False)

