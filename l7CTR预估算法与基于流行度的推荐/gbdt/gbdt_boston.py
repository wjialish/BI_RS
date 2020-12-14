# 数据集给出了影响房价的一些指标，比如犯罪率，房产税等，最后给出了房价
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
# 波士顿房价回归数据集
boston = load_boston()
# 划分训练集和数据集
X_train,X_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.1,random_state=0)
# n_estimators=500 分类器设置500个
clf = GradientBoostingRegressor(n_estimators=500,learning_rate=0.01,max_depth=4,min_samples_split=2,loss='ls')
clf.fit(X_train,y_train)
print('GBDT回归MSE：',mean_squared_error(y_test,clf.predict(X_test)))
# train_score_ 代表参差/误差，会发现随着分类器的不断增加，斜率越来越小，参差值会越来越小；当决策树/分类器取到100的时候，效果最好，随着决策树再增加，会发现效果不再明显
print('每次训练的得分记录：', clf.train_score_)
print('各特征重要程度：',clf.feature_importances_)

# 每次训练，增加新的Cart树，带来的训练得分变化
# train_score_:表示在样本集上每次迭代以后的对应损失函数值
plt.plot(np.arange(500),clf.train_score_,'b-')
plt.show()