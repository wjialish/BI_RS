'''
criteo_ctr数据集
展示广告CTR预估比赛
欧洲大型重定向广告公司Criteo的互联网广告数据集（4000万训练样本，500万测试样本）
原始数据：https://labs.criteo.com/2013/12/download-terabyte-click-logs/
small_train.txt 和 small_test.txt文件
（FFM数据格式，200条记录）
'''

import xlearn as xl

#创建FFM模型
ffm_model = xl.create_ffm()

#设置训练集和测试集
ffm_model.setTrain('small_train.txt')
ffm_model.setValidate('small_test.txt')

#设置参数，任务为二分类，学习旅为0.2 正则项lambda:0.002 评估指标 accuracy
param = {'task':'binary','lr':0.2,'lambda':0.002,'metric':'acc'}

#FFM训练，并输出模型
ffm_model.fit(param,'model.out')

#设置测试集，将输出结果转换为0-1
ffm_model.setTest('small_test.txt')
ffm_model.setSigmoid()

#使用训练好的FFM模型进行预测，输出到output.txt
ffm_model.predict('model.out','output.txt')

'''
----------------------------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \ 
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.40 Version --
----------------------------------------------------------------------------------------------

[------------] xLearn uses 8 threads for training task.
[ ACTION     ] Read Problem ...
[------------] First check if the text file has been already converted to binary format.
[------------] Binary file (small_train.txt.bin) found. Skip converting text to binary.
[------------] First check if the text file has been already converted to binary format.
[------------] Binary file (small_test.txt.bin) found. Skip converting text to binary.
[------------] Number of Feature: 9991
[------------] Number of Field: 18
[------------] Time cost for reading problem: 0.00 (sec)
[ ACTION     ] Initialize model ...
[------------] Model size: 5.56 MB
[------------] Time cost for model initial: 0.01 (sec)
[ ACTION     ] Start to train ...
[------------] Epoch      Train log_loss       Test log_loss       Test Accuarcy     Time cost (sec)
[   10%      ]     1            0.593632            0.536397            0.770000                0.00
[   20%      ]     2            0.533430            0.547050            0.770000                0.00
[   30%      ]     3            0.517541            0.534357            0.770000                0.00
[   40%      ]     4            0.505166            0.536103            0.770000                0.00
[   50%      ]     5            0.497740            0.533910            0.775000                0.00
[   60%      ]     6            0.483727            0.533486            0.775000                0.00
[   70%      ]     7            0.472459            0.529431            0.775000                0.00
[   80%      ]     8            0.464706            0.533023            0.770000                0.00
[   90%      ]     9            0.457136            0.531808            0.770000                0.00
[  100%      ]    10            0.449487            0.537230            0.770000                0.00
[ ACTION     ] Early-stopping at epoch 7, best Accuarcy: 0.775000
[ ACTION     ] Start to save model ...
[------------] Model file: model.out
[------------] Time cost for saving model: 0.01 (sec)
[ ACTION     ] Finish training
[ ACTION     ] Clear the xLearn environment ...
[------------] Total time cost: 0.03 (sec)
----------------------------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \ 
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.40 Version --
----------------------------------------------------------------------------------------------

[------------] xLearn uses 8 threads for prediction task.
[ ACTION     ] Load model ...
[------------] Load model from model.out
[------------] Loss function: cross-entropy
[------------] Score function: ffm
[------------] Number of Feature: 9991
[------------] Number of K: 4
[------------] Number of field: 18
[------------] Time cost for loading model: 0.01 (sec)
[ ACTION     ] Read Problem ...
[------------] First check if the text file has been already converted to binary format.
[------------] Binary file (small_test.txt.bin) found. Skip converting text to binary.
[------------] Time cost for reading problem: 0.00 (sec)
[ ACTION     ] Start to predict ...
[------------] The test loss is: 0.529431
[ ACTION     ] Clear the xLearn environment ...
[------------] Total time cost: 0.01 (sec)
'''
