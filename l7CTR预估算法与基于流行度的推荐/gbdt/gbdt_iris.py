from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 使用GBDT进行分类
clf = GradientBoostingClassifier(n_estimators=50,learning_rate=1.0,max_depth=1,random_state=0)
score = cross_val_score(clf,X,y,cv=3)
print('GBDT准确率： %0.41f' %score.mean())

'''
 learning rate:0.1  GBDT准确率： 0.97385620915032677924472181985038332641125
 learning rate:1.0  GBDT准确率： 0.96037581699346397101635375292971730232239
'''