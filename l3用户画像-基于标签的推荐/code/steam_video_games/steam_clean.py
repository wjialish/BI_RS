import numpy as np 
import pandas as pd 
import random

path = './steam-200k.csv'
df = pd.read_csv(path, header = None, names = ['UserID', 'Game', 'Action', 'Hours', 'Not Needed'])
# 数据探索
print('显示前5条数据')
print(df.head())
print('显示数据大小')
print(df.shape)
# 创建Hours_Played字段，替代原有的Action和Hours，0表示仅购买，大于0表示购买且游戏时长
df['Hours_Played'] = df['Hours'].astype('float32')
# 如果字段Action=purchase，并且Hours=1.0，将设置Hours_Played=0
df.loc[(df['Action'] == 'purchase') & (df['Hours'] == 1.0), 'Hours_Played'] = 0
print(df['Hours_Played'])
print('增加了Hours_Played字段后，数据大小')
print(df.shape)

# 对数据从小到大进行排序, df下标也会发生变化
df.UserID = df.UserID.astype('int')
df = df.sort_values(['UserID', 'Game', 'Hours_Played'], ascending=True)

# 删除重复项，并保留最后一项出现的项（因为最后一项是用户游戏时间，第一项为购买）
clean_df = df.drop_duplicates(['UserID', 'Game'], keep = 'last')
# 去掉不用的列：Action, Hours, Not Needed
clean_df = clean_df.drop(['Action', 'Hours', 'Not Needed'], axis = 1)
print('删除重复项后的数据集：')
print(clean_df)
print(clean_df.head(0))

# 探索下数据集的特征
n_users = len(clean_df.UserID.unique())
n_games = len(clean_df.Game.unique())
print('数据集中包含了 {0} 玩家，{1} 游戏'.format(n_users, n_games))

# 矩阵的稀疏性
sparsity = clean_df.shape[0] / float(n_users * n_games)
print('用户行为矩阵的稀疏性（填充比例）为{:.2%} '.format(sparsity))
