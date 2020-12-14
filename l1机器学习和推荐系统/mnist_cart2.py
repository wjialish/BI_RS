import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

#导入knn算法，决策树，逻辑斯蒂回归
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from IPython.display import display

#导入数字图片
#特征数据
X = []
#目标数据
y =[]
#一共有10个文件夹(数字0-9)，每个有500张图片
#图片命名格式为：0_1.bmp
for i in range(10):
    for j in range(1,501):
        #读取图片
        digit = plt.imread('./digits/%d/%d_%d.bmp'%(i,i,j))
        X.append(digit)
        y.append(i)

#把列表转成数组
X = np.array(X)
y = np.array(y)
#查看数组形状
X.shape

#随机显示一张图片
index = np.random.randint(0,5000,size=1)[0]
digit = X[index]
#设置画布宽为1，高为1
plt.figure(figsize=(1,1))
#显示颜色为gray
plt.imshow(digit,cmap='gray')
print(y[index])

#拆分数据：训练数据和测试数据
from sklearn.model_selection import train_test_split

#测试数据占比为0.1
#一共有5000张照片，那么用来做测试的有500张
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                test_size=0.1)

X_train.shape

# 784个像素---->784个属性-----> 数字不一样
X.reshape(5000,-1).shape

#使用决策树,深度为50
tree = DecisionTreeClassifier(max_depth=50)
#训练模型
tree.fit(X_train.reshape(4500,-1),y_train)
#对训练后的模型进行评分
tree.score(X_test.reshape(500,-1),y_test)


#可视化
#画布大小10行10列
#每行高为1，每列宽为1.5
plt.figure(figsize=(10*1,10*1.5))
for i in range(30):
    #绘制子图
    axes = plt.subplot(10,10,i+1)
    #测试数据为500张，绘制其中的30张
    axes.imshow(X_test[i],cmap='gray')
    #添加标题
    t = y_test[i]
    p = y_[i]
    axes.set_title('True:%d\nPred:%d'%(t,p))
    #不显示坐标刻度
    axes.axis('off')