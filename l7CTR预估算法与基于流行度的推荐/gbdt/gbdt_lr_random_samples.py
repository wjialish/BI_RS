
# 随机生成8万个样本，用于二分类训练
import numpy as np
np.random.seed(10)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

n_estimator = 10
# 生成样本集, 8万个样本，每个样本20个特征
X,y = make_classification(n_samples=80000,n_features=20)
print(X)
# 将样本集分为测试集和训练集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)
X_train,X_train_lr,y_train,y_train_lr = train_test_split(X_train,y_train,test_size=0.5)

# 直接使用LR进行预测
lr = LogisticRegression(n_jobs=4,C=0.1,penalty='l1机器学习和推荐系统')
lr.fit(X_train,y_train)
y_pred_lr = lr.predict_proba(X_test)[:,1]
fpr_lr,tpr_lr , _ = roc_curve(y_test,y_pred_lr)

# 基于随机森林的监督变换
rf = RandomForestClassifier(max_depth=3,n_estimators=n_estimator)
rf.fit(X_train,y_train)


# 直接使用RF进行预测
y_pred_rf = rf.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, thresholds_skl = roc_curve(y_test,y_pred_rf)


# 得到One-hot编码
rf_enc = OneHotEncoder(categories='auto')
rf_enc.fit(rf.apply(X_train))

# 使用One-hot编码作为特征，训练LR
rf_lr = LogisticRegression(solver='lbfgs',max_iter=1000)
rf_lr.fit(rf_enc.transform(rf.apply(X_train_lr)),y_train_lr)

# 使用RF+LR进行预测
y_pred_rf_lr =rf_lr.predict_proba(rf_enc.transform(rf.apply(X_test)))[:,1]
fpr_rf_lr,tpr_rf_lr,_ = roc_curve(y_test,y_pred_rf_lr)



# 基于GBDT监督变换
gbdt = GradientBoostingClassifier(max_depth=3,n_estimators=n_estimator)
gbdt.fit(X_train,y_train)

# 直接使用gbdt进行预测
y_pred_gbdt = gbdt.predict_proba(X_test)[:,1]
fpr_gbdt,tpr_gbdt, thresholds_skl= roc_curve(y_test,y_pred_gbdt)

# 得到one-hot编码
gbdt_enc = OneHotEncoder(categories='auto')
temp =gbdt.apply(X_train)
np.set_printoptions(threshold=np.inf)
gbdt_enc.fit(gbdt.apply(X_train)[:,:,0])
#使用One-hot编码作为特征，训练LR
gbdt_lr = LogisticRegression(solver='lbfgs',max_iter=1000)
gbdt_lr.fit(gbdt_enc.transform(gbdt.apply(X_train_lr)[:,:,0]),y_train_lr)

# 使用GBDT+LR进行预测
y_pred_gbdt_lr = gbdt_lr.predict_proba(gbdt_enc.transform(gbdt.apply(X_test)[:,:,0]))[:,1]
fpr_gbdt_lr,tpr_gbdt_lr, _ = roc_curve(y_test,y_pred_gbdt_lr)

plt.figure(1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_rf,tpr_rf, label = 'RF')
plt.plot(fpr_rf_lr,tpr_rf_lr,label ='RF+LR')
plt.plot(fpr_lr,tpr_lr,label ='LR')
plt.plot(fpr_gbdt,tpr_gbdt,label ='GBDT')
plt.plot(fpr_gbdt_lr,tpr_gbdt_lr,label ='GBDT_LR')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# 将左上角放大显示
plt.figure(2)
plt.xlim(0,0.2)
plt.ylim(0.8,1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_rf,tpr_rf, label = 'RF')
plt.plot(fpr_rf_lr,tpr_rf_lr,label ='RF+LR')
plt.plot(fpr_lr,tpr_lr,label ='LR')
plt.plot(fpr_gbdt,tpr_gbdt,label ='GBDT')
plt.plot(fpr_gbdt_lr,tpr_gbdt_lr,label ='GBDT_LR')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
