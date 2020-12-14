from surprise import Dataset,Reader
from surprise import KNNBasic,KNNWithMeans,KNNWithZScore,KNNBaseline
from surprise.model_selection import KFold
from surprise import accuracy

reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./ratings.csv', reader=reader)
# trainset = data.build_full_trainset()

# ItemCF 计算得分
# 取最相似的用户计算时，只取最相似的k个
# KNNWithMeans
algo = KNNWithMeans(k=50, sim_options={'user_based': False, 'verbose': 'True'})

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
# k 值越大，泛化结果越好，但是时间也越长
for trainset,testset in kf.split(data):
    # 训练并测试
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RSME
    accuracy.rmse(predictions,verbose=True)
    accuracy.mae(predictions,verbose=True)

# 196这个用户对于302这个电影的预测分数是怎么样的
uid = str(196)
iid = str(302)
#输出uid对iid的预测结果，原来的实际值r_ui是4分
pred = algo.predict(uid,iid,r_ui=4,verbose=True)


'''
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8556
MAE:  0.6546
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8560
MAE:  0.6543
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8575
MAE:  0.6556
user: 196        item: 302        r_ui = 4.00   est = 4.19   {'actual_k': 50, 'was_impossible': False}
'''






# KNNBasic
algo = KNNBasic(k=50, sim_options={'user_based': False, 'verbose': 'True'})

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
# k 值越大，泛化结果越好，但是时间也越长
for trainset,testset in kf.split(data):
    # 训练并测试
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RSME
    accuracy.rmse(predictions,verbose=True)
    accuracy.mae(predictions,verbose=True)

# 196这个用户对于302这个电影的预测分数是怎么样的
uid = str(196)
iid = str(302)
#输出uid对iid的预测结果，原来的实际值r_ui是4分
pred = algo.predict(uid,iid,r_ui=4,verbose=True)



'''
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8972
MAE:  0.6901
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8947
MAE:  0.6881
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8948
MAE:  0.6890
user: 196        item: 302        r_ui = 4.00   est = 3.80   {'actual_k': 50, 'was_impossible': False}
'''


# KNNWithZScore
algo = KNNWithZScore(k=50, sim_options={'user_based': False, 'verbose': 'True'})

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
# k 值越大，泛化结果越好，但是时间也越长
for trainset,testset in kf.split(data):
    # 训练并测试
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RSME
    accuracy.rmse(predictions,verbose=True)
    accuracy.mae(predictions,verbose=True)

# 196这个用户对于302这个电影的预测分数是怎么样的
uid = str(196)
iid = str(302)
#输出uid对iid的预测结果，原来的实际值r_ui是4分
pred = algo.predict(uid,iid,r_ui=4,verbose=True)



'''
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8598
MAE:  0.6564
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8580
MAE:  0.6550
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8579
MAE:  0.6554
user: 196        item: 302        r_ui = 4.00   est = 4.02   {'actual_k': 50, 'was_impossible': False}
'''




# KNNBaseline
algo = KNNBaseline(k=50, sim_options={'user_based': False, 'verbose': 'True'})

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
# k 值越大，泛化结果越好，但是时间也越长
for trainset,testset in kf.split(data):
    # 训练并测试
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RSME
    accuracy.rmse(predictions,verbose=True)
    accuracy.mae(predictions,verbose=True)

# 196这个用户对于302这个电影的预测分数是怎么样的
uid = str(196)
iid = str(302)
#输出uid对iid的预测结果，原来的实际值r_ui是4分
pred = algo.predict(uid,iid,r_ui=4,verbose=True)


'''
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8541
MAE:  0.6530
Estimating biases using als...
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8499
MAE:  0.6509
Estimating biases using als...
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8521
MAE:  0.6522
user: 196        item: 302        r_ui = 4.00   est = 4.07   {'actual_k': 50, 'was_impossible': False}
'''