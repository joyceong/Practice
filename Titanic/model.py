#-*- coding: UTF-8 -*-

import pandas
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, f_classif # 引入feature_selection观察每一个特征的重要程度

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

#-----数据处理
def dataProcessing():
    titanic = pandas.read_csv('train.csv')
    # print titanic.describe()
    
    #------处理Age&Sex
    titanic['Age'] =titanic['Age'].fillna(titanic['Age'].median()) # 对缺失值用中值填充
    titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
    titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
    
    #-----处理Embarked字段
    titanic['Embarked'] = titanic['Embarked'].fillna('S') # 用最多的填充空值
    titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0 
    titanic.loc[titanic['Embarked'] == 'C','Embarked'] = 1
    titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2 
    #-----提取特征
    titanic['Familysize'] = titanic['SibSp'] + titanic['Parch'] # 家庭人数
    titanic['NameLength'] = titanic['Name'].apply(lambda x : len(x)) # 名字长度
    titles = titanic['Name'].apply(get_title)
    # print pandas.value_counts(titles)
    title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Dr":5, "Rev":6, "Col":7, "Major":8, "Mlle":9, "Countess":10, "Ms":11, "Lady":12, "Jonkheer":13, "Don":14, "Mme":15, "Capt":16, "Sir":17}
    for k,v in title_mapping.items():
        titles[titles == k] = v
        # print (pandas.value_counts(titles))
    titanic["titles"] = titles # 添加title             
    return titanic

def get_title(name):
    title_reserch = re.search('([A-Za-z]+)\.', name)
    if title_reserch:
        return title_reserch.group(1)
    return ""

def dataVisualization(titanic): 
    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Familysize', 'NameLength', 'titles'] # 用到的特征
    # print len(predictors)
    selector = SelectKBest(f_classif, k=5)
    selector.fit(titanic[predictors], titanic['Survived'])
    scores = -np.log10(selector.pvalues_)
    plt.bar(left=np.arange(len(predictors)), height=scores) # left,height类型都是ndarray
    plt.xticks(np.arange(len(predictors)), predictors, rotation='vertical')
    plt.show()


#------Linear Regression
def linearRegression(predictors, titanic):
    alg = LinearRegression()
    kf = KFold(titanic.shape[0], n_folds=3, shuffle=False, random_state=1)
    predictions = []

    for train, test in kf:
        # print 'train:', train, '\ntest:',test
        train_predictors = (titanic[predictors].iloc[train,:])
        # print train_predictors
        train_target = titanic['Survived'].iloc[train]
        alg.fit(train_predictors, train_target)
        test_prediction = alg.predict(titanic[predictors].iloc[test,:])
        # print test_prediction
        predictions.append(test_prediction)
    predictions = np.concatenate(predictions, axis=0)
    predictions[predictions >.5] = 1
    predictions[predictions <=.5] = 0
    accuracy = sum(predictions[predictions == titanic['Survived']]) / len(predictions)
    return accuracy

#-----Logistic Regression
def logisticRegression(predictors, titanic):
    titanic = dataProcessing()
    alg = LogisticRegression(random_state=1)
    scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=3)
    print 'logistic regression:',scores.mean()

def randomForestClassifier(predictors, titanic):
    alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
    kf = cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
    scores = cross_validation.cross_val_score(alg,titanic[predictors],titanic['Survived'],cv=kf)
    print 'random forest:',scores.mean()

def boostingClassifier(predictors,titanic):
    algorithas = [
    [GradientBoostingClassifier(random_state=1,n_estimators=25,max_depth=3),predictor],
    [LogisticRegression(random_state=1),predictors]]
    kf = KFold(titanic.shape[0],n_folds=3,random_state=1)
    predictions = []
    for train, test in kf:
        train_target = titanic['Survived'].iloc[train]
        full_test_predictions = []
        for alg, predictors in algorithas:
            alg.fit(titanic[predictors].iloc[train,:],train_target)
            test_prediction = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))
            full_test_predictions.append(test_prediction)
        test_predictions.append(test_prediction)
        test_predictions[test_predictions >.5] = 1
        test_predictions[test_predictions <=.5] = 0
        predictions.append(test_predictions)
    predictions = np.concatenate(predictions, axis=0)
    accuracy = sum(predictions[predictions == titanic['Survived']])/ len(predictions)
    print accuracy

def deepLearning(predictors, titanic):
    train_x = titanic[predictors]
    label = titanic['Survived']
    train_y = np.zeros((2,len(label)))
    train_y[0,:] = label
    train_y[1,:] = np.ones((len(label))) - label
    model = Sequential()
    model.add(Dense(input_dim=len(predictors), units=100, activation='sigmoid'))
    model.add(Dense(units=100,activation='sigmoid'))
    model.add(Dense(units=2, activation='softmax'))
    model.compile(loss='mse', optimizer=SGD(lr=0.1),metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=100, epochs=20)
    train_results = model.evaluate(train_x, train_y, batch_size=100)
    print 'Deep Learning(Training):', train_results
    

def main():
    print 'aloha'
    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] # 用到的特征
    titanic = dataProcessing()
    # print titanic.shape
    dataVisualization(titanic)
    linearRegression(predictors, titanic)
    logisticRegression(predictors, titanic)
    randomForestClassifier(predictors, titanic)
    deepLearning(predictors, titanic)

if __name__ == '__main__':
    main()
    print 'aloha'

