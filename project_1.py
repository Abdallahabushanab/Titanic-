import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')

train.isnull().sum()
test.isnull().sum()
# All missing values from train data Age = 177, Cabin = 687, Embarked =2
# All missing values from train data Age = 86, Cabin = 327

# We will remove the Name and Ticket because there are no realted between survied and the name
train = train.drop(['Name','Ticket'], axis=1)
test = test.drop(['Name','Ticket'], axis=1)

# Let's understand our data more.
train.columns
test.columns
train.info()
test.info()
train.Fare.describe()

# Pclass, sex, SibSp, Parch and Fare are clear
# now let's deal with columns with missing values Cabin, Age, and Embarked.
age_under_1 = train[train.Age < 1]
train.Cabin.isnull().sum()/len(train)
train.Age.isnull().sum()/len(train)
train.Age.describe()

# Remove Cabin because  77% of the Values are missing
train = train.drop('Cabin', axis=1)
# for Age we have two options to deal with missing values (remove it or fill it)
plt.hist(train.Age)
train.Age.mode()
train.Age.median()
train.Age.mean()
# I will choose fill it with Mean, it not so differnts because it
# so close values between all mean, median, and mode.
# Also I think Embarked not important so we will remove it
train = train.drop('Embarked', axis=1)
'''train.dropna(subset= ['Embarked'], inplace=True) '''

train = train.fillna(train['Age'].mean())
train = pd.get_dummies(train)
train_dummy = pd.get_dummies(train)

train.columns
train.Sex_female.sum()/ len(train)
train.Sex_male.sum()/ len(train)
train.Survived.sum()

plt.scatter(train.SibSp, train.Survived)
train_1 = train[train.Survived == 1] 
train_0 = train[train.Survived == 0] 

x = pd.pivot_table(train_1,index='SibSp',columns='Pclass', values='Survived',
               aggfunc='sum')
y = pd.pivot_table(train_0,index='SibSp',columns='Pclass', values='Survived',
               aggfunc='count')
Percentage = x/y
train.groupby(['Survived'])['Sex_female','Sex_male'].sum()
train_1.groupby(['Survived','Parch'])['Survived'].sum()
train_0.groupby(['Survived','Parch'])['Survived'].count()

train.groupby(['Survived'])['Fare'].mean()

# test editing 
test = test.drop('Cabin', axis=1)
test = test.drop('Embarked', axis=1)
test = test.fillna(train['Age'].mean())
test = pd.get_dummies(test)

''' so it is binary problem need to slove so we will use one of the these 
algorthim :
Logistic Regression
k-Nearest Neighbors
Decision Trees
Support Vector Machine
Naive Bayes '''

train_p = train.Survived
train =  train.drop('Survived', axis = 1)
gender_submission = gender_submission.drop('PassengerId', axis =1 )
gender_submission.columns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
resulte = []


''' fit, fit_transform, predict, StandardScaler, predict_proba '''

# logistic Regression
log_model = LogisticRegression(solver = 'newton-cg', random_state=0, C = 10.0)
log_model.fit(train,train_p)
p_pred = log_model.predict_proba(test)
y_pred = log_model.predict(test)
score_ = accuracy_score(gender_submission, y_pred)
conf_m = confusion_matrix(gender_submission,y_pred)
report = classification_report(gender_submission, y_pred)
log_model.coef_
log_model.intercept_
resulte.append(score_)

# Decision Tree
tree_model = DecisionTreeClassifier(criterion = 'gini', splitter='best',
                                    random_state=0)
tree_model.fit(train, train_p)
p_pred = tree_model.predict_proba(test)
y_pred = tree_model.predict(test)
score_ = accuracy_score(gender_submission, y_pred)
conf_m = confusion_matrix(gender_submission, y_pred)
report = classification_report(gender_submission, y_pred)
tree_model.classes_
tree_model.n_outputs_
resulte.append(score_)


# Support Vector Machine

svc_model = SVC(C=1.0, degree = 2, kernel = 'linear')
svc_model.fit(train, train_p)
# p_pred = svc_model.predict_proba(test)
y_pred = svc_model.predict(test)
score_ = accuracy_score(gender_submission, y_pred)
conf_m = confusion_matrix(gender_submission, y_pred)
report = classification_report(gender_submission, y_pred)
svc_model.class_weight_
svc_model.classes_
resulte.append(score_)

# Naive Bayes 

navie_model = GaussianNB()
navie_model.fit(train, train_p)
y_pred = navie_model.predict(test)
p_pred = navie_model.predict_proba(test)
score_ = accuracy_score(gender_submission, y_pred)
conf_m = confusion_matrix(gender_submission, y_pred)
report = classification_report(gender_submission, y_pred)
navie_model.theta_
navie_model.sigma_
navie_model.class_count_
navie_model.class_prior_
navie_model.get_params(deep = True)
navie_model.predict_log_proba(test)
resulte.append(score_)
algo = ['logistic Regression', 'Decision Tree', 'Support Vector Machine',
        'Naive Bayes' ]

plt.scatter(algo, resulte)

# as we saw the support vector machine is the best model to predict our problem
test.columns
y_pred = pd.DataFrame(y_pred, columns = ['Survived'])
y_pred.insert(0,'PassengerId' ,test.PassengerId)
y_pred.reset_index(drop=True, inplace = True)
y_pred.to_csv(r'C:\Users\cne family\Desktop\Kaggle\Tatianc\y_pred.csv')


