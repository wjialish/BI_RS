import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.inputs import SparseFeat,get_feature_names
from sklearn.model_selection import train_test_split
import numpy as np
from deepctr.models import DeepFM
from sklearn.metrics import mean_squared_error


#数据加载
data = pd.read_csv('movielens_sample.txt')
sparse_features = ['movie_id','user_id','gender','age','occupation','zip']
target = ['rating']

#对标签进行编码
for feature in sparse_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature])

# 计算每个特征中的 不同特征值的个数
fixlen_feature_columns = [SparseFeat(feature, data[feature].nunique()) for feature in sparse_features]
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 将数据集切分为训练集和测试集
#data = np.loadtxt('movielens_sample.txt',dtype=object)
train, test = train_test_split(data,test_size=0.2)
# print(train)
# print('------------------')
# print(test)
# np.savetxt('movielens_smaple_train.txt',train,delimiter=',')
# np.savetxt('movielens_smaple_test.txt',test,delimiter=',')
train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}

#使用DeepFM进行训练
model = DeepFM(linear_feature_columns,dnn_feature_columns,task='regression')
# adam 一种优化算法，和SGD
model.compile('adam','mse',metrics=['mse'],)
history = model.fit(train_model_input,train[target].values,batch_size=256,epochs=1,verbose=True,validation_split=0.2)



# 使用DeepFM进行预测
pre_ans = model.predict(test_model_input,batch_size=256)
print(pre_ans)

#输出RMSE或MSE
mse = round(mean_squared_error(test[target].values,pre_ans),4)
rmse = mse ** 0.5
print('test RMSE',rmse)