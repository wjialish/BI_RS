# 词云展示
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
# from PIL import Image
import numpy as np
# from lxml import etree
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns

# 生成词云
def create_word_cloud(f):
	print('根据词频，开始生成词云!')
	# f = remove_stop_words(f)
	cut_text = word_tokenize(f)
	#print(cut_text)
	cut_text = " ".join(cut_text)
	wc = WordCloud(
		max_words=100,
		width=2000,
		height=1200,
    )
	wordcloud = wc.generate(cut_text)
	# 写词云图片
	wordcloud.to_file("wordcloud.jpg")
	# 显示词云文件
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()

# 数据加载
market_basket =  pd.read_csv("F:\git_clone_code\RS-ML相关\L3\MarketBasket\Market_Basket_Optimisation.csv",header=None)
# print(market_basket.head())

market_basket_0 = market_basket.fillna(0)
# print(market_basket_0.head())

# 将同一条transaction的产品放入set,所有set组成list
transactions = []
for i in range(len(market_basket_0)):
    temp_set = set()
    for index, item in market_basket_0.loc[i].items():  
        if item != 0:
            temp_set.add(item)
    transactions.append(temp_set)
# print(transactions)

# 提取所有产品并按照索引依次堆叠组成list，索引组成list
indexid = []
item = []
for m in range(len(transactions)):
	for x in transactions[m]:
		indexid.append(m)
		item.append(x)

# 计算词频
fre = FreqDist(item)
print(fre.most_common(10))

# 生成list
list_fre = list(fre.most_common(10))
list_item =[]
list_count =[]
for x in list_fre:
	list_item.append(x[0])
	list_count.append(x[1])

# 频率分布图
fre.tabulate(10)
fre.plot(10)

# 饼图..
plt.pie(x = list_count, labels=list_item)
plt.show()

# 柱状图
plt.bar(list_item, list_count)
plt.show()
# 用Seaborn画柱状图
sns.barplot(list_item, list_count)
plt.show()

# 索引和产品重组dataframe
data = pd.DataFrame({'item':item, 'transaction': indexid})

# 读取item字段
items = " ".join(data['item'])
# print(items)

# 生成词云
create_word_cloud(items)
