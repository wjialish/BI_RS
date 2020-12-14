#使用TPOT自动机器学习工具对MNIST进行分类
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from tpot import TPOTClassifier

#加载数据
digits = load_digits()
data=digits.data
print(digits.target)
X_train, X_test, y_train, y_test = train_test_split(digits.data.astype(np.float64),
                                                    digits.target.astype(np.float64),train_size=0.75,test_size=0.25)

#TPOT采用一些遗传算法，来进行参数的选型
#
tpot = TPOTClassifier(generations=5,population_size=20,verbosity=2)
tpot.fit(X_train,y_train)
print(tpot.score(X_test,y_test))
tpot.export('tpot_mnist_pipeline.py')

# [0 1 2... 8 9 8]
# Optimization
# Progress: 33 % |███▎ | 40 / 120[04:10 < 04:17, 3.22
# s / pipeline]Generation
# 1 - Current
# best
# internal
# CV
# score: 0.9702509911223954
# Optimization
# Progress: 49 % |████▉ | 59 / 120[06:0
# 8 < 12: 58, 12.76
# s / pipeline]Generation
# 2 - Current
# best
# internal
# CV
# score: 0.9709971983569806
# Optimization
# Progress: 67 % |██████▋ | 80 / 120[07:54 < 04:0
# 8, 6.21
# s / pipeline]Generation
# 3 - Current
# best
# internal
# CV
# score: 0.9888678757450368
# Optimization
# Progress: 83 % |████████▎ | 100 / 120[10:15 < 02:55, 8.77
# s / pipeline]Generation
# 4 - Current
# best
# internal
# CV
# score: 0.9888678757450368
# Optimization
# Progress: 98 % |█████████▊ | 118 / 120[10:32 < 00:01, 1.09
# pipeline / s]Generation
# 5 - Current
# best
# internal
# CV
# score: 0.9888678757450368
#
# Best
# pipeline: KNeighborsClassifier(PolynomialFeatures(input_matrix, degree=2, include_bias=False, interaction_only=False),
#                                n_neighbors=2, p=2, weights=distance)
# 0.9822222222222222