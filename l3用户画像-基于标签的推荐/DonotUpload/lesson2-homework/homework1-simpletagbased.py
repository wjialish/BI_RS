%%time
# encoding=utf-8
'''
author:lyq
'''
from collections import defaultdict,Counter
import random
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import log
'''
思路：
1.导入数据，并通过 utn_middle 来分析是否需要进行数据集划分，具体分为以下几种情况：
  1.1 utn_middle小于5，说明每个用户的历史行为数据较少只推荐用户评分偏高的标签即可，目的是进一步丰富用户
      评分偏高的标签数据
  1.2 utn_middle介于5-10之间，每个用户有了一定的数据，划分数据集：9-1
  1.3 utn_middle大于10，每个用户的数据较为丰富，划分数据集：9-1,8-2,7-3
2.依据用户的历史行为数据中的标签倒排与每个标签对应的book的热门倒排来为用户推荐（过滤掉用户已观看的）
  2.1依据评分 score 的不同来对 simpletagbased 进行改进：
      不同 simpletagbased 的根本在于此处的评分 score
    2.1.1.simpletagbased ：更多倾向于推荐热门
        score = i_num*t_num 
    2.1.2.TagBasedTFIDF：对热门标签惩罚，降低热门权重,惩罚函数为t_punish=(log((1+tn_dict[tt]),10))
        score = i_num*t_num/t_punish
    2.1.3.TagBasedTFIDF++：对热门标签和热门物品都惩罚，惩罚函数分别为
        t_punish=(log((1+tn_dict[tt]),10))    i_punish=(log((1+in_dict[tt]),10))
        score = i_num*t_num/(t_punish*i_punish)
3.评价（准确率，召回率，F值，新品率）
4.通过改变测试集的大小来测试不同情况下的推荐过，并将不同测试集下的推荐结果汇集在一起可视化分析
'''

class simpletagbased:
    def __init__(self, user_data_path, tag_data_path, ut_dict, tun_dict, ti_dict, it_dict, ui_dict, iun_dict, 
                 u_top_ts,t_top_is, candidate1_file, candidates, test_num, test_users, all_users, all_items,
                 all_tags,utn_middle, topN):
        self.user_data_path = user_data_path
        self.tag_data_path = tag_data_path
        self.ut_dict = ut_dict
        self.tun_dict = tun_dict
        self.ti_dict = ti_dict
        self.it_dict = it_dict
        self.ui_dict = ui_dict
        self.iun_dict = iun_dict
        self.u_top_ts = u_top_ts
        self.t_top_is = t_top_is
        self.candidate1_file = candidate1_file
        self.candidates = candidates
        self.test_num = test_num
        self.test_users = test_users
        self.all_users = all_users
        self.all_items = all_items
        self.all_tags = all_tags
        self.utn_middle = utn_middle
        self.topN = topN 
            
    def data_pre_analysis(self):
        ud = open(self.user_data_path)
        for line in ud:
            line = line.strip('\n')
            userID, bookmarkID, tagID, timestamp = line.split('\t')
            if userID == 'userID':
                continue
            self.ut_dict.setdefault(userID,defaultdict(int))
            self.ut_dict[userID][tagID] += 1
            self.tun_dict.setdefault(tagID, set())
            self.tun_dict[tagID].add(userID)
            self.ti_dict.setdefault(tagID,defaultdict(int))
            self.ti_dict[tagID][bookmarkID] += 1
            self.it_dict.setdefault(bookmarkID,defaultdict(int))
            self.it_dict[bookmarkID][tagID] += 1
            self.ui_dict.setdefault(userID,set())
            self.ui_dict[userID].add(bookmarkID)
            self.iun_dict.setdefault(bookmarkID, set())
            self.iun_dict[bookmarkID].add(userID)
            self.all_users.add(userID)
            self.all_items.add(bookmarkID)
            self.all_tags.append(tagID)
        tags_dict = dict(Counter(STB.all_tags))
        self.utn_middle = sorted(tags_dict.values())[int(len(tags_dict.values())/2)]
        if self.utn_middle < 5:
            print('utn_middle=%d, 用户数据量普遍较少，采用全数据量训练'%self.utn_middle)
        
        for u in self.ut_dict.keys():
            t_dict = dict(self.ut_dict[u])
            t_tuple = sorted(t_dict.items(),key=lambda x:x[1],reverse=True)
            self.ut_dict[u]=t_tuple
        for t in self.ti_dict.keys():
            i_dict = dict(self.ti_dict[t])
            i_tuple = sorted(i_dict.items(), key=lambda x:x[1],reverse=True)
            self.ti_dict[t]=i_tuple

    def recommend(self):
        N1 = self.u_top_ts
        N2 = self.t_top_is
        N3 = self.topN
        tn_dict = {}
        in_dict = {}
        for t,us in self.tun_dict.items():
            tn_dict[t] = len(us)
        for i,us in self.iun_dict.items():
            in_dict[i] = len(us)
        self.test_users = random.sample(list(self.ut_dict.keys()),self.test_num)
        for u in self.test_users:
            for ts in self.ut_dict[u][:N1]:
                self.candidates.setdefault(u,{})
                tt,t_num = ts[0],ts[1]
                for i in self.ti_dict[tt][:N2]:
                    item,i_num = i[0],i[1]
                    if item in self.ui_dict[u]:
                        continue
                # 不同 simpletagbased 的根本在于此处的评分 score
                # 1.simpletagbased ：更多倾向于推荐热门
                #  score = i_num*t_num 
                # 2.TagBasedTFIDF：对热门标签惩罚，降低热门权重,惩罚函数为t_punish=(log((1+tn_dict[tt]),10))
                #  score = i_num*t_num/t_punish
                # 3.TagBasedTFIDF++：对热门标签和热门物品都惩罚，惩罚函数分别为
                #  t_punish=(log((1+tn_dict[tt]),10))    i_punish=(log((1+in_dict[tt]),10))
                #   score = i_num*t_num/(t_punish*i_punish)
                    t_punish=(log((1+tn_dict[tt]),10))
                    i_punish=(log((1+in_dict[item]),10))
                    score = i_num*t_num/(t_punish*i_punish)
                    self.candidates[u][item]=score
            self.candidates[u] = sorted(self.candidates[u].items(),key=lambda x:x[1],reverse=True)[:N3]
            self.candidates[u] = dict((x,y) for x,y in self.candidates[u])
        
    def evaluate(self):
        watched_ts = []
        rec_ts = []
        rec_new_n = 0
        for u in self.test_users:
            watched_utn_dict = dict((x,y) for x,y in self.ut_dict[u])
            for t,n in watched_utn_dict.items():
                watched_ts += [t]*n
            rec = self.candidates[u]
            for i,n in rec.items():
                for t,n in self.it_dict[i].items():
                    rec_ts += [t]*n
                if i not in self.ui_dict[u]:
                    rec_new_n +=1
        hit = 0
        for wt in set(watched_ts):
            n_wt = watched_ts.count(wt)
            n_rt = rec_ts.count(wt)
            n = min(n_wt,n_rt)
            hit += n
        precision = hit/len(rec_ts)
        recall = hit/len(watched_ts)
        F = precision*recall*2/(precision+recall)
        new_rate = rec_new_n/(self.test_num*self.topN)
        print('precision: %.3f%%   recall: %.3f%%   F: %.3f%%   new_rate: %.3f%%'%
              (100*precision, 100*recall, 100*F, 100*new_rate))
        return precision,recall,new_rate
    
    gc.collect()
    
    
if __name__=='__main__':
    p_list,r_list,n_list,tn_list = [],[],[],[]
    for tn in tqdm(list(range(10,30,10))):
        STB = simpletagbased(user_data_path='D:/MLdata/delicious_data/user_taggedbookmarks-timestamps.dat',
                             tag_data_path='D:/MLdata/delicious_data/tags.dat',
                             ut_dict={},
                             tun_dict={},
                             ti_dict={},
                             it_dict={},
                             ui_dict={},
                             iun_dict={},
                             u_top_ts=5,
                             t_top_is=5,
                             candidate1_file='D:/MLdata/delicious_data/candidate1.csv',
                             candidates={},
                             test_num=tn,
                             test_users=[],
                             all_users=set(),
                             all_items=set(),
                             all_tags=[],
                             utn_middle=0,
                             topN=20)
        STB.data_pre_analysis()
        STB.recommend()
        precision,recall,new_rate = STB.evaluate()
        p_list.append(precision)
        r_list.append(recall)
        n_list.append(new_rate)
        tn_list.append(str(tn))
    
    plt.title('Result Analysis')  
    plt.plot(tn_list, p_list, color='green', label='precision')
    plt.plot(tn_list, r_list, color='blue', label='recall')
    plt.plot(tn_list, n_list, color='black', label='recall')
    plt.xlabel('test_num')
    plt.ylabel('rate')
    plt.legend()
    plt.show()