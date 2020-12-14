import pandas as pd

train = pd.read_csv('train.csv',index_col=0)
test = pd.read_csv('test.csv',index_col=0)

#print((train['Attrition']).value_counts())
'''
No     988
Yes    188
Name: Attrition, dtype: int64
'''

#处理Attrition字段
train['Attrition']=train['Attrition'].map(lambda x:1 if x=='Yes' else 0)

# 查看数据是否有空值
#print(train.isna().sum())
'''
user_id                     0
Age                         0
Attrition                   0
BusinessTravel              0
DailyRate                   0
Department                  0
DistanceFromHome            0
Education                   0
EducationField              0
EmployeeCount               0
EmployeeNumber              0
EnvironmentSatisfaction     0
Gender                      0
HourlyRate                  0
JobInvolvement              0
JobLevel                    0
JobRole                     0
JobSatisfaction             0
MaritalStatus               0
MonthlyIncome               0
MonthlyRate                 0
NumCompaniesWorked          0
Over18                      0
OverTime                    0
PercentSalaryHike           0
PerformanceRating           0
RelationshipSatisfaction    0
StandardHours               0
StockOptionLevel            0
TotalWorkingYears           0
TrainingTimesLastYear       0
WorkLifeBalance             0
YearsAtCompany              0
YearsInCurrentRole          0
YearsSinceLastPromotion     0
YearsWithCurrManager        0
dtype: int64
'''

# 去掉没用的列 员工号码，标准工时（=80）
#train = train.drop(['EmployeeNumber','StandardHours'],axis=1)
#test = test.drop(['EmployeeNumber','StandardHours'],axis=1)
train = train.drop(['EmployeeNumber','StandardHours','Over18','EmployeeCount'],axis=1)
test = test.drop(['EmployeeNumber','StandardHours','EmployeeCount','Over18'],axis=1)


#对于分类特征进行特征值编码
from sklearn.preprocessing import LabelEncoder
#attr=['Age','BusinessTravel','Department','Education','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']
attr=['Age','BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']

lbe_list=[]
for feature in attr:
    lbe=LabelEncoder()
    train[feature] = lbe.fit_transform(train[feature])
    test[feature]=lbe.transform(test[feature])
    lbe_list.append(lbe)


import ngboost as ng

from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(train.drop('Attrition',axis=1),train['Attrition'],test_size=0.2,random_state=42)

model = ng.NGBClassifier(n_estimators=1000,
                         learning_rate=0.01,
                         verbose=True,
                         verbose_eval=50)


model.fit(X_train,y_train)

predict = model.pred_param(test)
#predict = model.predict(test)

#predict = model.predict_proba(test)[:,1]

#predict = model.pred_dist(test)
print(predict)

test['Attrition'] = predict
test['Attrition']=test['Attrition'].map(lambda x:1 if x>=0.5 else 0)
test[['Attrition']].to_csv('ngb2_submission.csv')


