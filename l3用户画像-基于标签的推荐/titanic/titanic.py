import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import cross_val_score

#数据加载
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
#数据探索
# 查看train_data信息
'''
pd.set_option('display.max_columns', None) #显示所有列
print('查看数据信息：列名、非空个数、类型等')
print(train_data.info())
print('-'*30)
print('查看数据摘要')
print(train_data.describe())
print('-'*30)
print('查看离散数据分布')
print(train_data.describe(include=['O']))
print('-'*30)
print('查看前5条数据')
print(train_data.head())
print('-'*30)
print('查看后5条数据')
print(train_data.tail())
'''

#使用平均年龄来填充年龄中的nan值
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
#使用票价的均值来填充票价中的nan值
train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

print(train_data['Embarked'].value_counts())
#使用登录最多的港口来填充登录港口的nan值
train_data['Embarked'].fillna('S',inplace=True)
test_data['Embarked'].fillna('S',inplace=True)


#特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
print('训练集特征值',train_features)
train_labels = train_data['Survived']
print('train_labels',train_labels)
test_features = test_data[features]
print('测试集特征值',test_features)

#非数值化特征数值化
decv=DictVectorizer(sparse=False)
train_features=decv.fit_transform(train_features.to_dict(orient='record'))
print('decv.feature_names_',decv.feature_names_)
test_features=decv.transform(test_features.to_dict(orient='record'))

# 构造ID3决策树
# entropy 熵
clf = DecisionTreeClassifier(criterion='entropy')
# 决策树训练
clf.fit(train_features,train_labels)
# 决策树预测
pred_lables=clf.predict(test_features)
# 得到决策树准确率（基于训练集）
acc_decision_tree = round(clf.score(train_features,train_labels),6)
print(u'score准确率为 %0.4lf' % acc_decision_tree)

#使用K折交叉验证，统计决策树准确率
print(np.mean(cross_val_score(clf,train_features,train_labels,cv=10)))


'''
decv.feature_names_ ['Age', 'Embarked=C', 'Embarked=Q', 'Embarked=S', 'Fare', 'Parch', 'Pclass', 'Sex=female', 'Sex=male', 'SibSp']
score准确率为 0.9820
0.7768374758824197
'''
