#引入包
import pandas as pd

data=pd.read_csv('train.csv',encoding='gbk')
#print(data.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
None
'''

print(data.describe())
'''
 PassengerId    Survived      Pclass  ...       SibSp       Parch        Fare
count(数量)   891.000000  891.000000  891.000000  ...  891.000000  891.000000  891.000000
mean（平均值）    446.000000    0.383838    2.308642  ...    0.523008    0.381594   32.204208
std（标准差）     257.353842    0.486592    0.836071  ...    1.102743    0.806057   49.693429
min（最小值）       1.000000    0.000000    1.000000  ...    0.000000    0.000000    0.000000
25% (第一四分位数）    223.500000    0.000000    2.000000  ...    0.000000    0.000000    7.910400
50%（中位数）     446.000000    0.000000    3.000000  ...    0.000000    0.000000   14.454200
75%(第一四分位数)   668.500000    1.000000    3.000000  ...    1.000000    0.000000   31.000000
max (最大值)    891.000000    1.000000    3.000000  ...    8.000000    6.000000  512.329200

[8 rows x 7 columns]
'''

#查看离散数据类型的分布
print(data.describe(include=['O']))
'''
                                            Name   Sex  Ticket Cabin Embarked
count                                        891   891     891   204      889
unique                                       891     2     681   147        3
top     Kelly, Miss. Anna Katherine "Annie Kate"  male  347082    G6        S
freq                                           1   577       7     4      644
'''

#查看前五条数据
print(data.head(5))

#查看后五条数据
print(data.tail(5))

#查看Embarked字段中取值的分布数量
print(data['Embarked'].value_counts())
'''
[5 rows x 12 columns]
S    644
C    168
Q     77
'''