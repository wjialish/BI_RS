#随机梯度下降分类器,这个分类器有一个好处是能够高效地处理非常大的数据集
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score

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


#采用Z-Score规范化
ss=preprocessing.StandardScaler()
train_ss_x=ss.fit_transform(train_x)
test_ss_x=ss.transform(test_x)

#创建SGD分类器
#random_state参数，因为这个分类器有一定的随机性，所以它需一个随机种子
sgd=SGDClassifier(random_state=32)
sgd.fit(train_ss_x,train_y)
predict_y=sgd.predict(test_ss_x)
print('SGD准确率： %0.4lf' % accuracy_score(predict_y,test_y))


# (1797, 64)
# [[ 0.  0.  5. 13.  9.  1.  0.  0.]
#  [ 0.  0. 13. 15. 10. 15.  5.  0.]
#  [ 0.  3. 15.  2.  0. 11.  8.  0.]
#  [ 0.  4. 12.  0.  0.  8.  8.  0.]
#  [ 0.  5.  8.  0.  0.  9.  8.  0.]
#  [ 0.  4. 11.  0.  1. 12.  7.  0.]
#  [ 0.  2. 14.  5. 10. 12.  0.  0.]
#  [ 0.  0.  6. 13. 10.  0.  0.  0.]]
# 0
# SGD准确率： 0.9511


print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(sgd,train_ss_x,train_y,cv=10)))

# cross_val_score准确率为 0.9556