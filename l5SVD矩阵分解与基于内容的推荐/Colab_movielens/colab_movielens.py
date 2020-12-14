from surprise import SVD,SVDpp,NMF
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import KFold
from surprise import accuracy
#from google.colab import drive
import os
import time
#drive.mount('content/drive')


#os.chdir('content/drive/My Drive/Colab Notebooks')


time1 = time.time()

#数据读取
reader = Reader(line_format='user item rating timestamp',sep=',',skip_lines=1)
data = Dataset.load_from_file('ratings.csv',reader=reader)
train_set = data.build_full_trainset()

# 使用funkSVD
algo = SVD(biased=False)

# 定义K折交叉验证迭代器
kf = KFold(n_splits=3)
for trainset,testset in kf.split(data):
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions,verbose=True)

uid = str(196)
iid = str(302)
# 输出uid对iid的预测结果
pred = algo.predict(uid,iid,r_ui=4,verbose=True)
time2 = time.time()
print(time2 - time1)


'''
/Users/wenjiali/anaconda3/bin/python3.7 /lwj/AI/kaikeba/PyCharmProjects/RS/Recommended_System_l6/colab_movielens.py
RMSE: 0.8725
RMSE: 0.8732
RMSE: 0.8719
user: 196        item: 302        r_ui = 4.00   est = 3.89   {'was_impossible': False}
143.6582248210907
'''
