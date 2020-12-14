import pandas as pd
import numpy as np
from efficient_apriori import apriori

market_basket = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
print(market_basket.head())

market_basket_0 = market_basket.fillna(0)
print(market_basket_0.head())

# 将同一条transaction的产品放入set,所有set组成list
transactions = []
for i in range(len(market_basket_0)):
    temp_set = set()
    for index, item in market_basket_0.loc[i].items():  
        if item != 0:
            temp_set.add(item)
    transactions.append(temp_set)
# print(transactions)

# efficient_apriori方法
print('*'*20,'efficient_apriori方法','*'*80)
itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.3)
print('频繁项集：', itemsets)
print('关联规则：', rules)


# mlxtend方法
print('*'*20,'mlxtend方法','*'*80)
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
# 提取所有产品并按照索引依次堆叠组成list，索引组成list
indexid = []
item = []
for m in range(len(transactions)):
	for x in transactions[m]:
		indexid.append(m)
		item.append(x)

# 索引和产品重组dataframe
data = pd.DataFrame({'item':item, 'transaction': indexid})

hot_encode_data = data.groupby(['transaction','item'])['item'].count().unstack().reset_index().fillna(0).set_index('transaction')

frequent_itemsets = apriori(hot_encode_data, min_support=0.02, use_colnames = True)
rules = association_rules(frequent_itemsets, metric = 'lift', min_threshold=0.5)

print("频繁项集：", frequent_itemsets)
print("关联规则：", rules[ (rules['lift'] >= 1.2) & (rules['confidence'] >= 0.3) ])

