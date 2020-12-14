import pandas as pd
import time

# 数据加载
data = pd.read_csv('./BreadBasket_DMS.csv')
# 统一小写
data['Item'] = data['Item'].str.lower()
# 去掉none项
data = data.drop(data[data.Item == 'none'].index)

# 采用efficient_apriori工具包
def rule1():
	from efficient_apriori import apriori
	start = time.time()
	# 得到一维数组orders_series，并且将Transaction作为index, value为Item取值
	orders_series = data.set_index('Transaction')['Item']
	# 将数据集进行格式转换
	transactions = []
	temp_index = 0
	for i, v in orders_series.items():
		if i != temp_index:
			temp_set = set()
			temp_index = i
			temp_set.add(v)
			transactions.append(temp_set)
		else:
			temp_set.add(v)
	
	# 挖掘频繁项集和频繁规则
	itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.5)
	print('频繁项集：', itemsets)
	print('关联规则：', rules)
	end = time.time()
	print("用时：", end-start)



def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
# 采用mlxtend.frequent_patterns工具包
def rule2():
	from mlxtend.frequent_patterns import apriori
	from mlxtend.frequent_patterns import association_rules
	pd.options.display.max_columns=100
	start = time.time()
	hot_encoded_df=data.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
	hot_encoded_df = hot_encoded_df.applymap(encode_units)
	frequent_itemsets = apriori(hot_encoded_df, min_support=0.02, use_colnames=True)
	rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
	print("频繁项集：", frequent_itemsets)
	print("关联规则：", rules[ (rules['lift'] >= 1) & (rules['confidence'] >= 0.5) ])
	#print(rules['confidence'])
	end = time.time()
	print("用时：", end-start)

#rule1()

'''
频繁项集： {1: {('alfajores',): 344, ('bread',): 3096, ('brownie',): 379, ('cake',): 983, ('coffee',): 4528, ('cookies',): 515, ('farm house',): 371, ('hot chocolate',): 552, ('juice',): 365, ('medialuna',): 585, ('muffin',): 364, ('pastry',): 815, ('sandwich',): 680, ('scandinavian',): 275, ('scone',): 327, ('soup',): 326, ('tea',): 1350, ('toast',): 318, ('truffles',): 192}, 2: {('bread', 'cake'): 221, ('bread', 'coffee'): 852, ('bread', 'pastry'): 276, ('bread', 'tea'): 266, ('cake', 'coffee'): 518, ('cake', 'tea'): 225, ('coffee', 'cookies'): 267, ('coffee', 'hot chocolate'): 280, ('coffee', 'juice'): 195, ('coffee', 'medialuna'): 333, ('coffee', 'pastry'): 450, ('coffee', 'sandwich'): 362, ('coffee', 'tea'): 472, ('coffee', 'toast'): 224}}
关联规则： [{cake} -> {coffee}, {cookies} -> {coffee}, {hot chocolate} -> {coffee}, {juice} -> {coffee}, {medialuna} -> {coffee}, {pastry} -> {coffee}, {sandwich} -> {coffee}, {toast} -> {coffee}]
用时： 0.16760516166687012
'''

print('-'*100)
rule2()

'''
频繁项集：      support                 itemsets
0   0.036348              (alfajores)
1   0.327134                  (bread)
2   0.040046                (brownie)
3   0.103867                   (cake)
4   0.478445                 (coffee)
5   0.054417                (cookies)
6   0.039201             (farm house)
7   0.058326          (hot chocolate)
8   0.038567                  (juice)
9   0.061813              (medialuna)
10  0.038462                 (muffin)
11  0.086116                 (pastry)
12  0.071851               (sandwich)
13  0.029057           (scandinavian)
14  0.034552                  (scone)
15  0.034446                   (soup)
16  0.142646                    (tea)
17  0.033601                  (toast)
18  0.020287               (truffles)
19  0.023352            (bread, cake)
20  0.090025          (bread, coffee)
21  0.029163          (bread, pastry)
22  0.028107             (bread, tea)
23  0.054734           (coffee, cake)
24  0.023774              (tea, cake)
25  0.028212        (coffee, cookies)
26  0.029586  (coffee, hot chocolate)
27  0.020604          (juice, coffee)
28  0.035186      (medialuna, coffee)
29  0.047549         (pastry, coffee)
30  0.038250       (coffee, sandwich)
31  0.049873            (tea, coffee)
32  0.023669          (coffee, toast)
关联规则：         antecedents consequents  antecedent support  consequent support  \
9            (cake)    (coffee)            0.103867            0.478445   
13        (cookies)    (coffee)            0.054417            0.478445   
15  (hot chocolate)    (coffee)            0.058326            0.478445   
16          (juice)    (coffee)            0.038567            0.478445   
18      (medialuna)    (coffee)            0.061813            0.478445   
20         (pastry)    (coffee)            0.086116            0.478445   
23       (sandwich)    (coffee)            0.071851            0.478445   
27          (toast)    (coffee)            0.033601            0.478445   

     support  confidence      lift  leverage  conviction  
9   0.054734    0.526958  1.101399  0.005039    1.102557  
13  0.028212    0.518447  1.083608  0.002177    1.083069  
15  0.029586    0.507246  1.060199  0.001680    1.058451  
16  0.020604    0.534247  1.116632  0.002152    1.119810  
18  0.035186    0.569231  1.189753  0.005612    1.210754  
20  0.047549    0.552147  1.154046  0.006347    1.164569  
23  0.038250    0.532353  1.112674  0.003873    1.115276  
27  0.023669    0.704403  1.472276  0.007592    1.764411  
用时： 0.6094517707824707

Process finished with exit code 0

'''
