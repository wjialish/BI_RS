%%time
# encoding=utf-8
'''
author:lyq
about: data_analysis_pre of titanic data and prediction
'''



from tpot import TPOTClassifier
import gc
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter
import seaborn as sns
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from tpot import TPOTClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split,cross_val_score
class titanic_predict:
    
    '''
    异常值可视化分析：
    1.数据列信息：
    PassengerId:乘客ID
    Survived:是否获救，用1表示获救,用0表示没有获救
    Pclass:乘客等级，“1”表示Upper，“2”表示Middle，“3”表示Lower
    Name:乘客姓名
    Sex:性别
    Age:年龄
    SibSp:乘客在船上的配偶数量或兄弟姐妹数量）
    Parch:乘客在船上的父母或子女数量
    Ticket:船票信息
    Fare:票价
    Cabin:房号
    Embarked:表示乘客上船的码头距离泰坦尼克出发码头的距离，数值越大表示距离越远
    '''
    
    def __init__(self, train_data_path, test_data_path, train_data, test_data, tt_data, miss_rate,
                 g, ps, v):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.train_data = train_data
        self.test_data = test_data
        self.tt_data = tt_data
        self.miss_rate = miss_rate
        self.g = g
        self.ps = ps
        self.v = v
        
    def data_pre_analysis(self):
        self.tt_data = pd.read_csv(self.train_data_path)
        self.test_data = pd.read_csv(self.test_data_path)
        miss_v_statisic = self.tt_data.isnull().sum()
        miss_v_statisic_test = self.test_data.isnull().sum()
        print('训练集缺失值分析如下:\n',miss_v_statisic)
        print('测试集缺失值分析如下:\n',miss_v_statisic_test)
        '''
        缺失值处理：
        1.将缺失值数量超过一定比例的列删除
        2.数值列的缺失值采用min-max之间的随机数填充
        3.字符串列采用该列所有非空元素中随机抽取填充
        '''
        for c in miss_v_statisic.loc[miss_v_statisic!=0].index:
            miss_value_rate = len(self.tt_data.loc[self.tt_data[c].isnull()]
                                 )/len(self.tt_data.loc[self.tt_data[c].notnull()])
            if miss_value_rate > self.miss_rate:
                self.tt_data = self.tt_data.drop(c,axis=1)
            else:
                miss_v_index = self.tt_data.loc[self.tt_data[c].isnull()].index
                if c == 'Age':
                    min_age,max_age = min(self.tt_data.Age),max(self.tt_data.Age)
                    for mvi in miss_v_index:
                        # n = list(self.tt_data.columns).index(c)
                        # 方法2：self.tt_data.iloc[mvi,n] = random.randint(int(min_age),int(max_age))
                        self.tt_data.loc[mvi,c] = random.randint(int(min_age),int(max_age))
                elif c == 'Embarked':
                    for mvi in miss_v_index:
                        Embarked_notnull = list(self.tt_data.loc[self.tt_data.Embarked.notnull()].Embarked)
                        self.tt_data.loc[mvi,c] = random.sample(Embarked_notnull, 1)[0]
        miss_v_statisic = self.tt_data.isnull().sum()
        print('训练集缺失值处理的结果:\n',miss_v_statisic)
        if list(miss_v_statisic.loc[miss_v_statisic!=0]) == []:
            print('训练集缺失值处理完成')
            
        # 测试集缺失值处理
        for c in miss_v_statisic_test.loc[miss_v_statisic_test!=0].index:
            miss_value_rate_test = len(self.test_data.loc[self.test_data[c].isnull()]
                                 )/len(self.test_data.loc[self.test_data[c].notnull()])
            if miss_value_rate_test > self.miss_rate:
                self.test_data = self.test_data.drop(c,axis=1)
            elif c == 'Survived':
                continue
            else:
                miss_v_index_test = self.test_data.loc[self.test_data[c].isnull()].index
                if c == 'Age':
                    min_age,max_age = min(self.test_data.Age),max(self.test_data.Age)
                    for mvi in miss_v_index_test:
                        # n = list(self.tt_data.columns).index(c)
                        # 方法2：self.tt_data.iloc[mvi,n] = random.randint(int(min_age),int(max_age))
                        self.test_data.loc[mvi,c] = random.randint(int(min_age),int(max_age))
                elif c == 'Fare':
                    for mvi in miss_v_index_test:
                        Embarked_notnull = list(self.test_data.loc[self.test_data.Embarked.notnull()].Embarked)
                        self.test_data.loc[mvi,c] = random.sample(Embarked_notnull, 1)[0]
        miss_v_statisic_test = self.test_data.isnull().sum()
        print('测试集缺失值处理的结果:\n',miss_v_statisic_test)
        if list(miss_v_statisic_test.loc[miss_v_statisic_test!=0]) == []:
            print('测试集缺失值处理完成')               

        for c in self.tt_data.columns:
            col_type = type(list(self.tt_data[c])[0])
            if col_type==int or col_type==float:
                '''
                PassengerId,Survived,Pclass,Age,SibSp,Parch,Fare
                '''
                if c in ['Age','Fare']:
                    sns.distplot(self.tt_data[c],color="r",bins=30,kde=True)
                    plt.xlabel(c)
                    plt.ylabel('percentage')
                    plt.title(c+'-distribution')
                else:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.scatter(range(len(self.tt_data[c])), self.tt_data[c])
                    plt.title(c+'-distribution')
            if col_type == str:
                '''
                Name,Sex,Ticket,Cabin,Embarked
                '''
                c_dict = dict(Counter(self.tt_data[c]))
                c_tuple = sorted(c_dict.items(), key=lambda x:x[1], reverse=True)
                c_dict = dict((x,y) for x,y in c_tuple)  
                fig = plt.figure()
                ax = fig.add_subplot(111)
                if c in ['Ticket','Cabin']:
                    x = list(range(len(c_dict.keys())))
                    y = c_dict.values()
                    ax.scatter(x,y)
                    plt.xlabel(c)
                    plt.ylabel('level')
                    plt.title(c+'-distribution')
                else:
                    x = list(range(len(c_dict.keys())))
                    y = c_dict.values()
                    ax.scatter(x,y)
                    plt.xlabel(c)
                    plt.ylabel('counts')
                    plt.title(c+'-distribution')
                    
        # 特征对生存影响的可视化分析与基于方差贡献率的特征选择
        tt = self.tt_data
        children = len(tt.loc[(tt.Age<13)&(tt.Survived==1)])/len(tt.loc[(tt.Age<13)])
        f_teenager = len(tt.loc[(tt.Age<19)&(tt.Age>13)&(tt.Sex=='female')&(tt.Survived==1)]
                        )/len(tt.loc[(tt.Age<19)&(tt.Age>13)&(tt.Sex=='female')])
        m_teenager = len(tt.loc[(tt.Age<19)&(tt.Age>13)&(tt.Sex=='male')&(tt.Survived==1)]
                        )/len(tt.loc[(tt.Age<19)&(tt.Age>13)&(tt.Sex=='male')])
        f_youth = len(tt.loc[(tt.Age<35)&(tt.Age>19)&(tt.Sex=='female')&(tt.Survived==1)]
                     )/len(tt.loc[(tt.Age<35)&(tt.Age>19)&(tt.Sex=='female')])
        m_youth = len(tt.loc[(tt.Age<35)&(tt.Age>19)&(tt.Sex=='male')&(tt.Survived==1)]
                     )/len(tt.loc[(tt.Age<35)&(tt.Age>19)&(tt.Sex=='male')])
        middle_age = len(tt.loc[(tt.Age<50)&(tt.Age>35)&(tt.Survived==1)]
                        )/len(tt.loc[(tt.Age<50)&(tt.Age>35)])
        old = len(tt.loc[(tt.Age<50)&(tt.Survived==1)])/len(tt.loc[tt.Age<50])
        Sex_Age_Survived = [children, f_teenager, m_teenager, f_youth, m_youth, middle_age, old]
        age_list = ['child', 'f_teen', 'm_teen', 'f_youth', 'm_youth', 'mid_age', 'old']
        print(dict(zip(age_list, Sex_Age_Survived)))
        plt.scatter(age_list, Sex_Age_Survived)
        plt.xlabel('all-range-of-age')
        plt.ylabel('Survived-rate')
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.title('sex-age-survived-analysis')
        plt.show()

        F_0_S = len(tt.Fare.loc[(tt.Fare==0)&(tt.Survived==1)]
                             )/len(tt.Fare.loc[(tt.Fare==0)])
        F_50_S = len(tt.Fare.loc[(tt.Fare>0)&(tt.Fare<50)&(tt.Survived==1)]
                              )/len(tt.Fare.loc[(tt.Fare>0)&(tt.Fare<50)])
        F_100_S = len(tt.Fare.loc[(tt.Fare>40)&(tt.Fare<80)&(tt.Survived==1)]
                              )/len(tt.Fare.loc[(tt.Fare>50)&(tt.Fare<100)])
        F_150_S = len(tt.Fare.loc[(tt.Fare>100)&(tt.Fare<150)&(tt.Survived==1)]
                              )/len(tt.Fare.loc[(tt.Fare>100)&(tt.Fare<150)])
        F_250_S = len(tt.Fare.loc[(tt.Fare>150)&(tt.Fare<250)&(tt.Survived==1)]
                              )/len(tt.Fare.loc[(tt.Fare>150)&(tt.Fare<250)])
        F_350_S = len(tt.Fare.loc[(tt.Fare>250)&(tt.Fare<350)&(tt.Survived==1)]
                              )/len(tt.Fare.loc[(tt.Fare>250)&(tt.Fare<350)])
        F_max_S = len(tt.Fare.loc[(tt.Fare>350)&(tt.Survived==1)]
                              )/len(tt.Fare.loc[(tt.Fare>350)])
        F_S = [F_0_S, F_50_S, F_100_S, F_150_S, F_250_S, F_350_S, F_max_S]
        F_S_list = ['0', '50', '100', '150', '250', '350', 'max']
        print(dict(zip(F_S_list, F_S)))
        plt.scatter(F_S_list, F_S)
        plt.xlabel('all-range-of-Fare')
        plt.ylabel('Survived-rate')
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.title('fare-survived-analysis')
        plt.show()
        
        P_1_S = len(tt.Fare.loc[(tt.Pclass==1)&(tt.Survived==1)]
                              )/len(tt.Pclass.loc[tt.Pclass==1])
        P_2_S = len(tt.Fare.loc[(tt.Pclass==2)&(tt.Survived==1)]
                              )/len(tt.Pclass.loc[tt.Pclass==1])
        P_3_S = len(tt.Fare.loc[(tt.Pclass==3)&(tt.Survived==1)]
                              )/len(tt.Pclass.loc[tt.Pclass==1])
        P_S = [P_1_S, P_2_S, P_3_S]
        P_S_list = ['1', '2', '3']
        print(dict(zip(P_S_list, P_S)))
        plt.scatter(P_S_list, P_S)
        plt.xlabel('all-range-of-Pclass')
        plt.ylabel('Survived-rate')
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.title('Pclass-survived-analysis')
        plt.show()
        
        fs =[ Sex_Age_Survived,P_S,F_S]
        for f in fs:
            np.var(f)*len(f)
        print('特征选择结果:\n','Sex,Age,Fare,Pclass对生存的影响较大')
        

    def model_prediction(self):
        # 特征选择
        features = ['Pclass', 'Sex', 'Age', 'Fare']
        train_features = self.tt_data[features]
        train_labels = self.tt_data['Survived']
        test_features = self.test_data[features]
        print('特征值')
        print(train_features)
		
		# 该程序中并没有进行特征选择算法的撰写，人工筛选特征，所以 DictVectorizer输入的数据还是DataFrame
		# 通过 DictVectorizer将特征标签化
        dvec=DictVectorizer(sparse=False)
        train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
        print(dvec.feature_names_)
        
        # tpot  此处开始调用训练模型，此处使用TPOT，方法与其他的模型 DecisionTreeClassifier,xgb,LR等等使用方法一样
		# 1.设置模型参数导入模型
		# 2.fit拟合，预测，得出准确率
        tpot = TPOTClassifier(generations=self.g, population_size=self.ps, verbosity=self.v)
        tpot.fit(train_features, train_labels)
        test_features=dvec.transform(test_features.to_dict(orient='record'))
        pred_labels = tpot.predict(test_features)
        acc_tpot = tpot.score(train_features, train_labels)
        print(u'score准确率为 %.4lf' % acc_tpot)
        
        # CART决策树
        clf = DecisionTreeClassifier(criterion='gini')
        clf.fit(train_features, train_labels)
        pred_labels = clf.predict(test_features)
        acc_decision_tree = round(clf.score(train_features, train_labels), 6)
        print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))
        
	gc.collect()
		
if __name__=='__main__':
    TTP = titanic_predict(train_data_path='D:/MLdata/titanic_data/train.csv',
                          test_data_path='D:/MLdata/titanic_data/test.csv',
                          train_data=[],
                          test_data=[],
                          tt_data=[],
                          miss_rate=0.5,
                          g=5,
                          ps=20,
                          v=2)
    TTP.data_pre_analysis()
    TTP.model_prediction()