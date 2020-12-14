# 使用SimpleTagBased算法对Delicious2K数据进行推荐
# 原始数据集：https://grouplens.org/datasets/hetrec-2011/
# 数据格式：userID     bookmarkID     tagID     timestamp
import random
import math
import operator

class NormTagBased():
    #构造函数
    def __init__(self,filename):
        self.filename=filename
        self.loadData()
        self.randomlySplitData(0.2)
        self.initStat()
        self.testRecommend()



    #数据加载
    def loadData(self):
        print("开始加载数据......")
        filename=self.filename
        #保存了用户对item的tag
        self.records={}
        fi=open(filename)
        lineNum=0
        for line in fi:
            #print("数据加载前格式。。。")
            #print(line)
            lineNum+=1
            if lineNum==1:
                continue
            uid,iid,tag,timestamp=line.split('\t')
            #在数组中对应的坐标-1
            uid=int(uid)-1
            iid=int(iid)-1
            tag=int(tag)-1
            self.records.setdefault(uid,{})
            self.records[uid].setdefault(iid,[])
            self.records[uid][iid].append(tag)
            #print("数据加载后格式。。。")
            #print(line)
        fi.close()
        print("数据集大小为 %d."%(lineNum))




    #将数据集拆分为训练集和测试集
    def randomlySplitData(self,ratio,seed=100):
        random.seed(seed)
        self.train=dict()
        self.test=dict()
        for u in self.records.keys():
            for i in self.records[u].keys():
                if random.random()<ratio:
                    self.test.setdefault(u,{})
                    self.test[u].setdefault(i,[])
                    for t in self.records[u][i]:
                        self.test[u][i].append(t)
                else:
                    self.train.setdefault(u,{})
                    self.train[u].setdefault(i,[])
                    for t in self.records[u][i]:
                        self.train[u][i].append(t)

        print("训练集样本数 %d, 测试集样本数 %d " %(len(self.train),len(self.test)))






    #使用训练集，初始化user_tags,tag_items, user_items
    def initStat(self):
        records=self.train
        self.user_tags=dict()
        self.tag_items=dict()
        self.user_items=dict()
        self.tag_users=dict()
        for u,items in records.items():
            for i,tags in items.items():
                for tag in tags:
                    #print tag
                    #用户和tag的关系
                    self._addValueToMat(self.user_tags,u,tag,1)
                    #tag和item的关系
                    self._addValueToMat(self.tag_items,tag,i,1)
                    #用户和item的关系
                    self._addValueToMat(self.user_items,u,i,1)
                    #标签和用户的关系，某标签使用过的用户
                    self._addValueToMat(self.tag_users,tag,u,1)
        print("user_tags,tag_items,user_items初始化完成完成。")
        print("user_tag 大小 %d, tag_items大小 %d, user_items大小 %d tag_users大小 %d" % (len(self.user_tags),len(self.tag_items),len(self.user_items),len(self.tag_users)))






    # 设置矩阵 mat[index,item] = 1
    def _addValueToMat(self,mat,index,item,value=1):
        if index not in mat:
            mat.setdefault(index,{})
            mat[index].setdefault(item,value)
        else:
            if item not in mat[index]:
                mat[index].setdefault(item,value)
            else:
                mat[index][item]+=value







    # 使用测试集，计算准确率和召回率
    def precissionAndRecall(self,N):
        hit=0
        h_recall=0
        h_precision=0
        for user,items in self.test.items():
            if user not in self.train:
                continue
            # 获取Top-N推荐列表
            rank=self.recommand(user,N)
            for item,rui in rank:
                if item in items:
                    hit+=1
            h_recall+=len(items)
            h_precision+=N
        print("一共命中 %d 个，一共推荐 %d 个，用户设置Tag总数为 %d 个" %(hit,h_precision,h_recall))
        #返回准确率和召回率
        return (hit/(h_precision*1.0)),(hit/(h_recall*1.0))




    #对用户user推荐Top-N
    def recommand(self,user,N):
        recommend_items=dict()
        #对Item进行拆分，分数为所有的 （用户对某标签使用的次数wut,乘以商品被打上相同标签的次数wti）之和
        tagged_items=self.user_items[user]
        for tag, wut in self.user_tags[user].items():
            #print(self.user_tags[user])
            #print(tag)
            #print(wut)
            #norm_wut= wut/len(self.user_tags[user])
            for item,wti in self.tag_items[tag].items():
                #print(self.tag_items[tag])
                #print(wti)
                #norm_wti=wti/len(self.tag_items[tag])
                if item in tagged_items:
                    continue
                #print('wut=%s,wti=%s' %(wut,wti))
                if item not in recommend_items:
                    #recommend_items[item] = (wut/tag) * (wti/item)
                    #recommend_items[item] = wut * wti
                    #recommend_items[item] = norm_wut * norm_wti
                    recommend_items[item] = (wut/len(self.user_tags[user])) * (wti/len(self.tag_users[tag]))
                else:
                    #recommend_items[item] += (wut/tag) * (wti/item)
                    #recommend_items[item] += wut * wti
                    #recommend_items[item] += norm_wut * norm_wti
                    recommend_items[item] += (wut/len(self.user_tags[user])) * (wti/len(self.tag_users[tag]))
        return sorted(recommend_items.items(),key=operator.itemgetter(1),reverse=True)[0:N]





    # 使用测试集，对推荐结果进行评估
    def testRecommend(self):
        print("推荐结果评估")
        print("%3s %10s %10s" %('N',"精确率",'召回率'))
        for n in [5,10,20,30,40,50,60,70,80,90,100]:
            precision,recall=self.precissionAndRecall(n)
            print("%3d %10.3f%% %10.3f%%" %(n,precision*100,recall*100))



if __name__=='__main__':
    std=NormTagBased("./user_taggedbookmarks-timestamps.dat")



#tag_items[tag]
# 开始加载数据......
# 数据集大小为 437594.
# 训练集样本数 1860, 测试集样本数 1793
# user_tags,tag_items,user_items初始化完成完成。
# user_tag 大小 1860, tag_items大小 36884, user_items大小 1860
# 推荐结果评估
#   N        精确率        召回率
# 一共命中 72 个，一共推荐 8930 个，用户设置Tag总数为 20861 个
#   5      0.806%      0.345%
# 一共命中 103 个，一共推荐 17860 个，用户设置Tag总数为 20861 个
#  # 10      0.577%      0.494%





#tag_users[tag]
# 开始加载数据......
# 数据集大小为 437594.
# 训练集样本数 1860, 测试集样本数 1793
# user_tags,tag_items,user_items初始化完成完成。
# user_tag 大小 1860, tag_items大小 36884, user_items大小 1860 tag_users大小 36884
# 推荐结果评估
#   N        精确率        召回率
# 一共命中 81 个，一共推荐 8930 个，用户设置Tag总数为 20861 个
#   5      0.907%      0.388%
# 一共命中 114 个，一共推荐 17860 个，用户设置Tag总数为 20861 个
#  10      0.638%      0.546%
# 一共命中 181 个，一共推荐 35720 个，用户设置Tag总数为 20861 个
#  20      0.507%      0.868%
# 一共命中 213 个，一共推荐 53580 个，用户设置Tag总数为 20861 个
#  30      0.398%      1.021%
# 一共命中 254 个，一共推荐 71440 个，用户设置Tag总数为 20861 个
#  40      0.356%      1.218%
# 一共命中 278 个，一共推荐 89300 个，用户设置Tag总数为 20861 个
#  50      0.311%      1.333%
# 一共命中 308 个，一共推荐 107160 个，用户设置Tag总数为 20861 个
#  60      0.287%      1.476%
# 一共命中 338 个，一共推荐 125020 个，用户设置Tag总数为 20861 个
#  70      0.270%      1.620%
# 一共命中 365 个，一共推荐 142880 个，用户设置Tag总数为 20861 个
#  80      0.255%      1.750%
# 一共命中 396 个，一共推荐 160740 个，用户设置Tag总数为 20861 个
#  90      0.246%      1.898%
# 一共命中 430 个，一共推荐 178600 个，用户设置Tag总数为 20861 个
# 100      0.241%      2.061%

