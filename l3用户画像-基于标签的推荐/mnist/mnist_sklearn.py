#使用多种分类器进行mnist手写数字分类
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

#加载数据
digits=load_digits()
data=digits.data

#数据探索
print(data.shape)

#查看第一幅图像
print(digits.images[0])

#第一幅图像代表的数字含义
print(digits.target[0])

#将第一幅图像显示出来
# plt.gray()
# plt.imshow(digits.images[0])
# plt.show()

#分割数据，将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)
print(train_x.max())
print((train_x>1).sum())
print(train_x.shape[0]*train_x.shape[1])



#采用Z-Score规范化
ss=preprocessing.StandardScaler()
train_ss_x=ss.fit_transform(train_x)
test_ss_x=ss.transform(test_x)

#创建LR分类器
lr=LogisticRegression(solver='liblinear', multi_class='auto')#数据集比较小，使用liblinear,数据集大使用sag或者saga
lr.fit(train_ss_x,train_y)
predict_y=lr.predict(test_ss_x)
print('LR准确率： %0.4lf' % accuracy_score(predict_y,test_y))

# LR准确率： 0.9600


#创建线形 CART决策树分类器
cart=DecisionTreeClassifier()
cart.fit(train_ss_x,train_y)
predict_y=cart.predict(test_ss_x)
print('CART准确率： %0.4lf' % accuracy_score(predict_y,test_y))

# CART准确率： 0.8600

# 创建LDA分类器
lda=LinearDiscriminantAnalysis(n_components=2)
lda.fit(train_ss_x,train_y)
predict_y=lda.predict(test_ss_x)
print('LDA准确率： %0.4lf' % accuracy_score(predict_y,test_y))

# LDA准确率： 0.9378


#创建朴素贝叶斯分类器
NaiveBayes=BernoulliNB()
NaiveBayes.fit(train_ss_x,train_y)
predict_y=NaiveBayes.predict(test_ss_x)
print('NaiveBayes准确率： %0.4lf' % accuracy_score(predict_y,test_y))

# NaiveBayes准确率： 0.9378


#创建SVM分类器
svm=svm.SVC(kernel='rbf', C=1.0, gamma='auto')
svm.fit(train_ss_x,train_y)
predict_y=svm.predict(test_ss_x)
print('SVM准确率： %0.4lf' % accuracy_score(predict_y,test_y))


#创建KNN分类器
knn=KNeighborsClassifier()
knn.fit(train_ss_x,train_y)
predict_y= knn.predict(test_ss_x)
print('KNN准确率： %0.4lf' % accuracy_score(predict_y,test_y))



#创建AdaBoost分类器
#弱分类器
dt_stump=DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)
dt_stump.fit(train_ss_x,train_y)
#设置Adaboost迭代次数
n_estimators=500
adaboost=AdaBoostClassifier(base_estimator=dt_stump,n_estimators=n_estimators)
adaboost.fit(train_ss_x,train_y)
predict_y=adaboost.predict(test_ss_x)
print('AdaBoost准确率： %0.4lf' % accuracy_score(predict_y,test_y))



#创建XGBoost分类器
#强分类器
xgboost=XGBClassifier()
xgboost.fit(train_ss_x,train_y)
predict_y=xgboost.predict(test_ss_x)
print('XGBoost准确率： %0.4lf' % accuracy_score(predict_y,test_y))


# LR准确率： 0.9600
# CART准确率： 0.8467
# LDA准确率： 0.9378
# NaiveBayes准确率： 0.8822
# /Users/wenjiali/anaconda3/lib/python3.7/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
#   warnings.warn("Variables are collinear.")
# SVM准确率： 0.9867
# KNN准确率： 0.9756
# AdaBoost准确率： 0.9689
# XGBoost准确率： 0.9489