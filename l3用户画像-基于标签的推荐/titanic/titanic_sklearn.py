#使用多种分类器进行titanic生存预测
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression#逻辑回归
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier#决策树
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis#LDA线形判别分析
from sklearn.naive_bayes import BernoulliNB #伯努利朴素贝叶斯
from sklearn import svm #SVM
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.ensemble import AdaBoostClassifier#Adaboost
from xgboost import XGBClassifier #XGBoost
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier


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


# #采用Z-Score规范化
# ss=preprocessing.StandardScaler()
# train_ss_features=ss.fit_transform(train_features)
# test_ss_features=ss.transform(test_features)
train_ss_features=train_features
test_ss_features=test_features


#创建LR分类器
lr=LogisticRegression(solver='liblinear', multi_class='auto')#数据集比较小，使用liblinear,数据集大使用sag或者saga
lr.fit(train_ss_features,train_labels)
predict_y=lr.predict(test_ss_features)
acc_lr = round(lr.score(train_features,train_labels),6)
print('LR准确率： %0.4lf' % acc_lr)



#创建线形 CART决策树分类器
cart=DecisionTreeClassifier()
cart.fit(train_ss_features,train_labels)
predict_y=cart.predict(test_ss_features)
acc_cart = round(cart.score(train_features,train_labels),6)
print('CART准确率： %0.4lf' % acc_cart)



# 创建LDA分类器
lda=LinearDiscriminantAnalysis(n_components=2)
lda.fit(train_ss_features,train_labels)
predict_y=lda.predict(test_ss_features)
acc_lda = round(lda.score(train_features,train_labels),6)
print('LDA准确率： %0.4lf' % acc_lda)




#创建朴素贝叶斯分类器
NaiveBayes=BernoulliNB()
NaiveBayes.fit(train_ss_features,train_labels)
predict_y=NaiveBayes.predict(test_ss_features)
acc_NaivesBayes = round(NaiveBayes.score(train_features,train_labels),6)
print('NaiveBayes准确率： %0.4lf' % acc_NaivesBayes)



#创建SVM分类器
svm=svm.SVC(kernel='rbf', C=1.0, gamma='auto')
svm.fit(train_ss_features,train_labels)
predict_y=svm.predict(test_ss_features)
acc_svm = round(svm.score(train_features,train_labels),6)
print('SVM准确率： %0.4lf' % acc_svm)


#创建KNN分类器
knn=KNeighborsClassifier()
knn.fit(train_ss_features,train_labels)
predict_y= knn.predict(test_ss_features)
acc_knn = round(knn.score(train_features,train_labels),6)
print('KNN准确率： %0.4lf' % acc_knn)



#创建AdaBoost分类器
#弱分类器
dt_stump=DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)
dt_stump.fit(train_ss_features,train_labels)
#设置Adaboost迭代次数
n_estimators=500
adaboost=AdaBoostClassifier(base_estimator=dt_stump,n_estimators=n_estimators)
adaboost.fit(train_ss_features,train_labels)
predict_y=adaboost.predict(test_ss_features)
acc_adaboost = round(adaboost.score(train_features,train_labels),6)
print('AdaBoost准确率： %0.4lf' % acc_adaboost)



#创建XGBoost分类器
#强分类器
xgboost=XGBClassifier()
xgboost.fit(train_ss_features,train_labels)
predict_y=xgboost.predict(test_ss_features)
acc_xgboost = round(xgboost.score(train_features,train_labels),6)
print('XGBoost准确率： %0.4lf' % acc_xgboost)

#创建SGD分类器
#random_state参数，因为这个分类器有一定的随机性，所以它需一个随机种子
sgd=SGDClassifier(random_state=32)
sgd.fit(train_ss_features,train_labels)
predict_y=sgd.predict(test_ss_features)
acc_sgd = round(sgd.score(train_features,train_labels),6)
print('SGD准确率： %0.4lf' % acc_sgd)



'''
LR准确率： 0.8036
CART准确率： 0.9820
LDA准确率： 0.7991
NaiveBayes准确率： 0.7868
SVM准确率： 0.8900
KNN准确率： 0.8193
AdaBoost准确率： 0.9820
XGBoost准确率： 0.8721
SGD准确率： 0.7452
'''




'''
#采用Z-Score规范化 结果
LR准确率： 0.6319
CART准确率： 0.6162
LDA准确率： 0.6285
NaiveBayes准确率： 0.7868
SVM准确率： 0.6162
KNN准确率： 0.6341
AdaBoost准确率： 0.4085
XGBoost准确率： 0.6162
SGD准确率： 0.6431
'''