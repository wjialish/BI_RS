from surprise import Reader
from surprise import Dataset
from surprise import BaselineOnly,NormalPredictor
from surprise.model_selection import KFold
from surprise import accuracy
#数据读取
reader = Reader(line_format='user item rating timestamp',sep=',',skip_lines=1)
data = Dataset.load_from_file('./ratings.csv',reader=reader)
train_set = data.build_full_trainset()

#ALS优化
#bsl_options = {'method': 'als','n_epochs':5,'reg_u':12,'reg_i':5}
'''
'n_epochs':5, 迭代次数，默认是10
'reg_u':12, 用户的正则化参数，默认是15
'reg_i':5 物品的正则化参数，默认是10
'''
''''
Baseline算法ALS 优化
Estimating biases using als...
RMSE: 0.8643
Estimating biases using als...
RMSE: 0.8629
Estimating biases using als...
RMSE: 0.8645
user: 196        item: 302        r_ui = 4.00   est = 3.96   {'was_impossible': False}
'''

# SGD 优化
'''
参数：
reg: 代价函数的正则化参数默认0.02
learnng_rate:学习率，默认0.005
n_epochs, 迭代次数，默认是10
'''
#bsl_options ={'method':'sgd','n_epoch':5}
'''
Baseline算法SGD 优化
Estimating biases using sgd...
RMSE: 0.8650
Estimating biases using sgd...
RMSE: 0.8648
Estimating biases using sgd...
RMSE: 0.8638
user: 196        item: 302        r_ui = 4.00   est = 4.18   {'was_impossible': False}
'''


#algo = BaselineOnly(bsl_options=bsl_options)


#algo = BaselineOnly()
'''
Estimating biases using als...
RMSE: 0.8657
Estimating biases using als...
RMSE: 0.8662
Estimating biases using als...
RMSE: 0.8659
user: 196        item: 302        r_ui = 4.00   est = 4.19   {'was_impossible': False}
'''

algo = NormalPredictor()
'''
RMSE: 1.4326
RMSE: 1.4333
RMSE: 1.4316
user: 196        item: 302        r_ui = 4.00   est = 4.84   {'was_impossible': False}
'''

# 定义K折交叉验证迭代器，K=3
'''
交叉验证（Cross Validation）为CV。
基本思想：将原始数据进行分组，一部分作为训练集，另一部分作为测试集，首先用训练集对分类器进行训练，再利用验证集来测试训练
得到的模型，以此作为评价分类器的性能指标。
Kfold:
原始数据分成K组（一般是均分），将每个子集数据分别做一次验证集，其余的k-1组子集数据作为训练集，
这样会得到k个模型，用这k个模型最终的验证集的分类准确率的平均数作为此K-CV下分类器的性能指标。
StratifiedKFlod：
是K-Flod的变种，会返回stratified(分层的折叠)；每个小集合中，各个分类的样例比例大致和完整数据集中相同。

另一种交叉验证：

参见L2-titanic_clean.py
# 使用K折交叉验证 统计决策树准确率
print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))


两者区别
https://zhuanlan.zhihu.com/p/24825503
'''
kf = KFold(n_splits=3)
# k 值越大，泛化结果越好，但是时间也越长
for trainset,testset in kf.split(data):
    # 训练并测试
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RSME
    accuracy.rmse(predictions,verbose=True)


# 196这个用户对于302这个电影的预测分数是怎么样的
uid = str(196)
iid = str(302)
#输出uid对iid的预测结果，原来的实际值r_ui是4分
pred = algo.predict(uid,iid,r_ui=4,verbose=True)