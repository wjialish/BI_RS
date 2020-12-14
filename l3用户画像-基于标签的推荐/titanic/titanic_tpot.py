import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.model_selection import cross_val_score
from tpot import TPOTClassifier

#数据加载
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
#使用平均年龄来填充年龄中的nan值
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
#使用票价的均值来填充票价中的nan值
train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

#print(train_data['Embarked'].value_counts())
#使用登录最多的港口来填充登录港口的nan值
train_data['Embarked'].fillna('S',inplace=True)
test_data['Embarked'].fillna('S',inplace=True)


#特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
#print('训练集特征值',train_features)
train_labels = train_data['Survived']
#print('train_labels',train_labels)
test_features = test_data[features]
#print('测试集特征值',test_features)


#非数值化特征数值化
decv=DictVectorizer(sparse=False)
train_features=decv.fit_transform(train_features.to_dict(orient='record'))
print('decv.feature_names_',decv.feature_names_)
test_features=decv.transform(test_features.to_dict(orient='record'))



# 采用tpot算法
tpot=TPOTClassifier(generations=10,population_size=20,verbosity=3)
tpot.fit(train_features,train_labels)
pred_lables=tpot.predict(test_features)
print('测试集预测.....')
print(pred_lables)
#得到tpot准确率，基于训练集预测
acc_tpot=round(tpot.score(train_features,train_labels),6)
print(u'tpot算法预测score准确率为 %0.4lf' % acc_tpot)
print(tpot.score(train_features,train_labels))
tpot.export('tpot_titanic_pipeline.py')

'''
测试集预测.....
[0 0 0 0 0 0 0 0 1 0 0 0 1 0 1 1 0 0 1 0 0 1 1 1 1 0 1 0 1 0 0 0 1 0 1 0 0
 0 0 1 0 1 0 1 1 0 0 0 1 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 0 1 1 0 0 0
 1 0 0 1 0 1 1 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0
 0 1 1 1 0 0 1 1 1 1 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0
 1 0 1 0 0 1 0 0 1 1 1 1 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0 1 1 0 1
 0 1 0 0 0 0 0 0 0 1 0 1 1 0 0 1 1 0 1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 1 0 1 0
 1 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0 1 1 1 1 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 1
 0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 0 1 0 0 0 1 0 0
 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 1 0 0 0 1 0 1 0 0 1 0 1 1 0 1 0 0 0 1 1
 0 1 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0
 0 0 1 0 1 0 0 1 0 0 0]
tpot算法预测score准确率为 0.9282
'''


#使用K折交叉验证，统计tpot准确率
print(np.mean(cross_val_score(tpot,train_features,train_labels,cv=3)))





