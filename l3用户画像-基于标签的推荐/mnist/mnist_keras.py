#使用LeNet模型对Mnist手写数字进行识别
from keras.datasets import mnist
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

#数据加载
(train_x, train_y),(test_x,test_y) = mnist.load_data()

#输入数据为mnist数据集
train_x =train_x.reshape(train_x.shape[0],28,28,1)
test_x = test_x.reshape(test_x.shape[0],28,28,1)
train_x=train_x/255
test_x=test_x/255
train_y=keras.utils.to_categorical(train_y,10)
test_y=keras.utils.to_categorical(test_y,10)
#创建序贯模型
model = Sequential()
#第一层卷基层：6个卷积核，大小为5*5，relu激活函数
model.add(Conv2D(6,kernel_size=(5,5),activation='relu',input_shape=(28,28,1)))
# 第二层池化层： 最大池化
model.add(MaxPooling2D(pool_size=(2,2)))
# 第三层卷基层：16个卷积核，大小为5*5 relu激活函数
model.add(Conv2D(16,kernel_size=(5,5),activation='relu'))
# 第二层池化层：最大池化
model.add(MaxPooling2D(pool_size=(2,2)))
# 将参数进行扁平化，在LeNet5中称之为卷积层，实际上这一层是一维向量，和全连接层一样
model.add(Flatten())
model.add(Dense(120,activation='relu'))
# 全连接层，输出节点个数为84个
model.add(Dense(84,activation='relu'))
# 输出层，用softmax激活函数计算分类概率
model.add(Dense(10,activation='softmax'))
# 设置损失函数和优化器提醒
model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#传入数据进行训练
model.fit(train_x, train_y,batch_size=128,epochs=2, verbose=1,validation_data=(test_x,test_y))
#对结果进行评估
score=model.evaluate(test_x,test_y)
print('误差：%0.41f' %score[0])
print('准确率：',score[1])




# Train on 60000 samples, validate on 10000 samples
# Epoch 1/2
#
#   128/60000 [..............................] - ETA: 1:33 - loss: 2.3059 - accuracy: 0.1406
#   512/60000 [..............................] - ETA: 31s - loss: 2.2744 - accuracy: 0.1699
#   896/60000 [..............................] - ETA: 22s - loss: 2.2582 - accuracy: 0.1842
#  1280/60000 [..............................] - ETA: 18s - loss: 2.2303 - accuracy: 0.2352
#  1664/60000 [..............................] - ETA: 16s - loss: 2.1918 - accuracy: 0.2837
#  2048/60000 [>.............................] - ETA: 15s - loss: 2.1443 - accuracy: 0.3267
#  2432/60000 [>.............................] - ETA: 14s - loss: 2.0872 - accuracy: 0.3610
#  2816/60000 [>.............................] - ETA: 13s - loss: 2.0150 - accuracy: 0.3999
#  3200/60000 [>.............................] - ETA: 13s - loss: 1.9417 - accuracy: 0.4341
#  3456/60000 [>.............................] - ETA: 13s - loss: 1.8833 - accuracy: 0.4578
#  3840/60000 [>.............................] - ETA: 12s - loss: 1.7996 - accuracy: 0.4846
#  4224/60000 [=>............................] - ETA: 12s - loss: 1.7158 - accuracy: 0.5099
#  4608/60000 [=>............................] - ETA: 12s - loss: 1.6395 - accuracy: 0.5302
#  4992/60000 [=>............................] - ETA: 11s - loss: 1.5697 - accuracy: 0.5489
#  5248/60000 [=>............................] - ETA: 11s - loss: 1.5267 - accuracy: 0.5610
#  5504/60000 [=>............................] - ETA: 11s - loss: 1.4859 - accuracy: 0.5712
#  5888/60000 [=>............................] - ETA: 11s - loss: 1.4238 - accuracy: 0.5881
#  6144/60000 [==>...........................] - ETA: 11s - loss: 1.3887 - accuracy: 0.5980
#  6528/60000 [==>...........................] - ETA: 11s - loss: 1.3392 - accuracy: 0.6112
#  6912/60000 [==>...........................] - ETA: 11s - loss: 1.2886 - accuracy: 0.6259
#  7296/60000 [==>...........................] - ETA: 10s - loss: 1.2450 - accuracy: 0.6384
#  7680/60000 [==>...........................] - ETA: 10s - loss: 1.2081 - accuracy: 0.6495
#  8064/60000 [===>..........................] - ETA: 10s - loss: 1.1727 - accuracy: 0.6597
#  8448/60000 [===>..........................] - ETA: 10s - loss: 1.1412 - accuracy: 0.6683
#  8832/60000 [===>..........................] - ETA: 10s - loss: 1.1084 - accuracy: 0.6780
#  9216/60000 [===>..........................] - ETA: 10s - loss: 1.0789 - accuracy: 0.6861
#  9600/60000 [===>..........................] - ETA: 9s - loss: 1.0551 - accuracy: 0.6927
#  9984/60000 [===>..........................] - ETA: 9s - loss: 1.0277 - accuracy: 0.7000
# 10368/60000 [====>.........................] - ETA: 9s - loss: 1.0040 - accuracy: 0.7062
# 10752/60000 [====>.........................] - ETA: 9s - loss: 0.9797 - accuracy: 0.7133
# 11136/60000 [====>.........................] - ETA: 9s - loss: 0.9614 - accuracy: 0.7185
# 11520/60000 [====>.........................] - ETA: 9s - loss: 0.9408 - accuracy: 0.7243
# 11776/60000 [====>.........................] - ETA: 9s - loss: 0.9268 - accuracy: 0.7284
# 12160/60000 [=====>........................] - ETA: 9s - loss: 0.9088 - accuracy: 0.7337
# 12544/60000 [=====>........................] - ETA: 9s - loss: 0.8925 - accuracy: 0.7382
# 12928/60000 [=====>........................] - ETA: 9s - loss: 0.8768 - accuracy: 0.7426
# 13312/60000 [=====>........................] - ETA: 8s - loss: 0.8622 - accuracy: 0.7469
# 13696/60000 [=====>........................] - ETA: 8s - loss: 0.8456 - accuracy: 0.7518
# 14080/60000 [======>.......................] - ETA: 8s - loss: 0.8302 - accuracy: 0.7561
# 14464/60000 [======>.......................] - ETA: 8s - loss: 0.8153 - accuracy: 0.7603
# 14848/60000 [======>.......................] - ETA: 8s - loss: 0.8028 - accuracy: 0.7638
# 15232/60000 [======>.......................] - ETA: 8s - loss: 0.7903 - accuracy: 0.7675
# 15616/60000 [======>.......................] - ETA: 8s - loss: 0.7790 - accuracy: 0.7711
# 16000/60000 [=======>......................] - ETA: 8s - loss: 0.7678 - accuracy: 0.7744
# 16384/60000 [=======>......................] - ETA: 8s - loss: 0.7567 - accuracy: 0.7772
# 16768/60000 [=======>......................] - ETA: 8s - loss: 0.7450 - accuracy: 0.7805
# 17152/60000 [=======>......................] - ETA: 8s - loss: 0.7355 - accuracy: 0.7834
# 17536/60000 [=======>......................] - ETA: 7s - loss: 0.7255 - accuracy: 0.7863
# 17920/60000 [=======>......................] - ETA: 7s - loss: 0.7149 - accuracy: 0.7897
# 18304/60000 [========>.....................] - ETA: 7s - loss: 0.7052 - accuracy: 0.7925
# 18688/60000 [========>.....................] - ETA: 7s - loss: 0.6945 - accuracy: 0.7957
# 19072/60000 [========>.....................] - ETA: 7s - loss: 0.6848 - accuracy: 0.7986
# 19456/60000 [========>.....................] - ETA: 7s - loss: 0.6759 - accuracy: 0.8014
# 19840/60000 [========>.....................] - ETA: 7s - loss: 0.6674 - accuracy: 0.8036
# 20224/60000 [=========>....................] - ETA: 7s - loss: 0.6592 - accuracy: 0.8062
# 20608/60000 [=========>....................] - ETA: 7s - loss: 0.6517 - accuracy: 0.8082
# 20992/60000 [=========>....................] - ETA: 7s - loss: 0.6449 - accuracy: 0.8100
# 21376/60000 [=========>....................] - ETA: 7s - loss: 0.6368 - accuracy: 0.8122
# 21760/60000 [=========>....................] - ETA: 6s - loss: 0.6287 - accuracy: 0.8146
# 22016/60000 [==========>...................] - ETA: 6s - loss: 0.6239 - accuracy: 0.8161
# 22400/60000 [==========>...................] - ETA: 6s - loss: 0.6172 - accuracy: 0.8180
# 22784/60000 [==========>...................] - ETA: 6s - loss: 0.6107 - accuracy: 0.8200
# 23168/60000 [==========>...................] - ETA: 6s - loss: 0.6050 - accuracy: 0.8217
# 23552/60000 [==========>...................] - ETA: 6s - loss: 0.5982 - accuracy: 0.8235
# 23936/60000 [==========>...................] - ETA: 6s - loss: 0.5931 - accuracy: 0.8249
# 24320/60000 [===========>..................] - ETA: 6s - loss: 0.5863 - accuracy: 0.8269
# 24704/60000 [===========>..................] - ETA: 6s - loss: 0.5809 - accuracy: 0.8284
# 25088/60000 [===========>..................] - ETA: 6s - loss: 0.5749 - accuracy: 0.8302
# 25472/60000 [===========>..................] - ETA: 6s - loss: 0.5696 - accuracy: 0.8318
# 25856/60000 [===========>..................] - ETA: 6s - loss: 0.5638 - accuracy: 0.8334
# 26240/60000 [============>.................] - ETA: 6s - loss: 0.5585 - accuracy: 0.8351
# 26624/60000 [============>.................] - ETA: 6s - loss: 0.5530 - accuracy: 0.8368
# 27008/60000 [============>.................] - ETA: 5s - loss: 0.5479 - accuracy: 0.8383
# 27392/60000 [============>.................] - ETA: 5s - loss: 0.5434 - accuracy: 0.8397
# 27776/60000 [============>.................] - ETA: 5s - loss: 0.5386 - accuracy: 0.8410
# 28160/60000 [=============>................] - ETA: 5s - loss: 0.5340 - accuracy: 0.8422
# 28544/60000 [=============>................] - ETA: 5s - loss: 0.5289 - accuracy: 0.8437
# 28928/60000 [=============>................] - ETA: 5s - loss: 0.5248 - accuracy: 0.8449
# 29312/60000 [=============>................] - ETA: 5s - loss: 0.5210 - accuracy: 0.8460
# 29696/60000 [=============>................] - ETA: 5s - loss: 0.5172 - accuracy: 0.8472
# 30080/60000 [==============>...............] - ETA: 5s - loss: 0.5125 - accuracy: 0.8486
# 30464/60000 [==============>...............] - ETA: 5s - loss: 0.5081 - accuracy: 0.8501
# 30848/60000 [==============>...............] - ETA: 5s - loss: 0.5040 - accuracy: 0.8512
# 31232/60000 [==============>...............] - ETA: 5s - loss: 0.5005 - accuracy: 0.8523
# 31616/60000 [==============>...............] - ETA: 5s - loss: 0.4966 - accuracy: 0.8534
# 32000/60000 [===============>..............] - ETA: 5s - loss: 0.4933 - accuracy: 0.8542
# 32384/60000 [===============>..............] - ETA: 4s - loss: 0.4901 - accuracy: 0.8553
# 32768/60000 [===============>..............] - ETA: 4s - loss: 0.4864 - accuracy: 0.8565
# 33152/60000 [===============>..............] - ETA: 4s - loss: 0.4831 - accuracy: 0.8574
# 33536/60000 [===============>..............] - ETA: 4s - loss: 0.4797 - accuracy: 0.8583
# 33920/60000 [===============>..............] - ETA: 4s - loss: 0.4754 - accuracy: 0.8596
# 34304/60000 [================>.............] - ETA: 4s - loss: 0.4718 - accuracy: 0.8607
# 34688/60000 [================>.............] - ETA: 4s - loss: 0.4678 - accuracy: 0.8618
# 35072/60000 [================>.............] - ETA: 4s - loss: 0.4649 - accuracy: 0.8626
# 35456/60000 [================>.............] - ETA: 4s - loss: 0.4610 - accuracy: 0.8637
# 35840/60000 [================>.............] - ETA: 4s - loss: 0.4575 - accuracy: 0.8648
# 36224/60000 [=================>............] - ETA: 4s - loss: 0.4548 - accuracy: 0.8655
# 36608/60000 [=================>............] - ETA: 4s - loss: 0.4518 - accuracy: 0.8664
# 36992/60000 [=================>............] - ETA: 4s - loss: 0.4491 - accuracy: 0.8671
# 37376/60000 [=================>............] - ETA: 4s - loss: 0.4465 - accuracy: 0.8679
# 37760/60000 [=================>............] - ETA: 3s - loss: 0.4438 - accuracy: 0.8687
# 38144/60000 [==================>...........] - ETA: 3s - loss: 0.4410 - accuracy: 0.8695
# 38528/60000 [==================>...........] - ETA: 3s - loss: 0.4383 - accuracy: 0.8702
# 38912/60000 [==================>...........] - ETA: 3s - loss: 0.4361 - accuracy: 0.8710
# 39296/60000 [==================>...........] - ETA: 3s - loss: 0.4340 - accuracy: 0.8715
# 39680/60000 [==================>...........] - ETA: 3s - loss: 0.4310 - accuracy: 0.8724
# 40064/60000 [===================>..........] - ETA: 3s - loss: 0.4283 - accuracy: 0.8732
# 40448/60000 [===================>..........] - ETA: 3s - loss: 0.4254 - accuracy: 0.8742
# 40832/60000 [===================>..........] - ETA: 3s - loss: 0.4225 - accuracy: 0.8750
# 41216/60000 [===================>..........] - ETA: 3s - loss: 0.4198 - accuracy: 0.8759
# 41600/60000 [===================>..........] - ETA: 3s - loss: 0.4174 - accuracy: 0.8766
# 41984/60000 [===================>..........] - ETA: 3s - loss: 0.4150 - accuracy: 0.8773
# 42368/60000 [====================>.........] - ETA: 3s - loss: 0.4127 - accuracy: 0.8780
# 42752/60000 [====================>.........] - ETA: 3s - loss: 0.4102 - accuracy: 0.8787
# 43136/60000 [====================>.........] - ETA: 3s - loss: 0.4083 - accuracy: 0.8792
# 43520/60000 [====================>.........] - ETA: 2s - loss: 0.4060 - accuracy: 0.8798
# 43904/60000 [====================>.........] - ETA: 2s - loss: 0.4036 - accuracy: 0.8805
# 44288/60000 [=====================>........] - ETA: 2s - loss: 0.4015 - accuracy: 0.8810
# 44672/60000 [=====================>........] - ETA: 2s - loss: 0.3992 - accuracy: 0.8817
# 45056/60000 [=====================>........] - ETA: 2s - loss: 0.3970 - accuracy: 0.8824
# 45440/60000 [=====================>........] - ETA: 2s - loss: 0.3950 - accuracy: 0.8830
# 45824/60000 [=====================>........] - ETA: 2s - loss: 0.3928 - accuracy: 0.8837
# 46208/60000 [======================>.......] - ETA: 2s - loss: 0.3907 - accuracy: 0.8844
# 46592/60000 [======================>.......] - ETA: 2s - loss: 0.3889 - accuracy: 0.8850
# 46976/60000 [======================>.......] - ETA: 2s - loss: 0.3869 - accuracy: 0.8855
# 47360/60000 [======================>.......] - ETA: 2s - loss: 0.3846 - accuracy: 0.8861
# 47744/60000 [======================>.......] - ETA: 2s - loss: 0.3830 - accuracy: 0.8865
# 48128/60000 [=======================>......] - ETA: 2s - loss: 0.3808 - accuracy: 0.8871
# 48512/60000 [=======================>......] - ETA: 2s - loss: 0.3793 - accuracy: 0.8877
# 48896/60000 [=======================>......] - ETA: 1s - loss: 0.3773 - accuracy: 0.8883
# 49280/60000 [=======================>......] - ETA: 1s - loss: 0.3753 - accuracy: 0.8889
# 49664/60000 [=======================>......] - ETA: 1s - loss: 0.3738 - accuracy: 0.8893
# 50048/60000 [========================>.....] - ETA: 1s - loss: 0.3720 - accuracy: 0.8898
# 50432/60000 [========================>.....] - ETA: 1s - loss: 0.3699 - accuracy: 0.8904
# 50816/60000 [========================>.....] - ETA: 1s - loss: 0.3681 - accuracy: 0.8910
# 51200/60000 [========================>.....] - ETA: 1s - loss: 0.3663 - accuracy: 0.8915
# 51584/60000 [========================>.....] - ETA: 1s - loss: 0.3642 - accuracy: 0.8922
# 51968/60000 [========================>.....] - ETA: 1s - loss: 0.3622 - accuracy: 0.8927
# 52352/60000 [=========================>....] - ETA: 1s - loss: 0.3603 - accuracy: 0.8933
# 52736/60000 [=========================>....] - ETA: 1s - loss: 0.3587 - accuracy: 0.8938
# 53120/60000 [=========================>....] - ETA: 1s - loss: 0.3569 - accuracy: 0.8943
# 53504/60000 [=========================>....] - ETA: 1s - loss: 0.3555 - accuracy: 0.8946
# 53888/60000 [=========================>....] - ETA: 1s - loss: 0.3540 - accuracy: 0.8951
# 54272/60000 [==========================>...] - ETA: 1s - loss: 0.3526 - accuracy: 0.8954
# 54656/60000 [==========================>...] - ETA: 0s - loss: 0.3511 - accuracy: 0.8959
# 55040/60000 [==========================>...] - ETA: 0s - loss: 0.3497 - accuracy: 0.8962
# 55424/60000 [==========================>...] - ETA: 0s - loss: 0.3480 - accuracy: 0.8967
# 55808/60000 [==========================>...] - ETA: 0s - loss: 0.3464 - accuracy: 0.8972
# 56192/60000 [===========================>..] - ETA: 0s - loss: 0.3449 - accuracy: 0.8976
# 56576/60000 [===========================>..] - ETA: 0s - loss: 0.3433 - accuracy: 0.8980
# 56960/60000 [===========================>..] - ETA: 0s - loss: 0.3417 - accuracy: 0.8985
# 57344/60000 [===========================>..] - ETA: 0s - loss: 0.3401 - accuracy: 0.8990
# 57728/60000 [===========================>..] - ETA: 0s - loss: 0.3387 - accuracy: 0.8993
# 58112/60000 [============================>.] - ETA: 0s - loss: 0.3371 - accuracy: 0.8997
# 58496/60000 [============================>.] - ETA: 0s - loss: 0.3357 - accuracy: 0.9001
# 58880/60000 [============================>.] - ETA: 0s - loss: 0.3342 - accuracy: 0.9005
# 59264/60000 [============================>.] - ETA: 0s - loss: 0.3327 - accuracy: 0.9009
# 59648/60000 [============================>.] - ETA: 0s - loss: 0.3311 - accuracy: 0.9014
# 60000/60000 [==============================] - 11s 184us/step - loss: 0.3298 - accuracy: 0.9017 - val_loss: 0.1077 - val_accuracy: 0.9661
# Epoch 2/2
#
#   128/60000 [..............................] - ETA: 9s - loss: 0.0781 - accuracy: 0.9688
#   512/60000 [..............................] - ETA: 9s - loss: 0.0685 - accuracy: 0.9805
#   896/60000 [..............................] - ETA: 10s - loss: 0.1010 - accuracy: 0.9732
#  1280/60000 [..............................] - ETA: 10s - loss: 0.0983 - accuracy: 0.9734
#  1664/60000 [..............................] - ETA: 10s - loss: 0.1034 - accuracy: 0.9700
#  2048/60000 [>.............................] - ETA: 10s - loss: 0.1037 - accuracy: 0.9692
#  2432/60000 [>.............................] - ETA: 9s - loss: 0.1046 - accuracy: 0.9675
#  2816/60000 [>.............................] - ETA: 9s - loss: 0.1058 - accuracy: 0.9659
#  3200/60000 [>.............................] - ETA: 9s - loss: 0.1050 - accuracy: 0.9659
#  3584/60000 [>.............................] - ETA: 9s - loss: 0.1019 - accuracy: 0.9671
#  3968/60000 [>.............................] - ETA: 9s - loss: 0.1030 - accuracy: 0.9672
#  4352/60000 [=>............................] - ETA: 9s - loss: 0.1051 - accuracy: 0.9671
#  4736/60000 [=>............................] - ETA: 9s - loss: 0.1081 - accuracy: 0.9668
#  5120/60000 [=>............................] - ETA: 9s - loss: 0.1097 - accuracy: 0.9656
#  5504/60000 [=>............................] - ETA: 9s - loss: 0.1091 - accuracy: 0.9657
#  5888/60000 [=>............................] - ETA: 9s - loss: 0.1103 - accuracy: 0.9657
#  6272/60000 [==>...........................] - ETA: 9s - loss: 0.1081 - accuracy: 0.9659
#  6656/60000 [==>...........................] - ETA: 9s - loss: 0.1096 - accuracy: 0.9653
#  7040/60000 [==>...........................] - ETA: 9s - loss: 0.1114 - accuracy: 0.9653
#  7424/60000 [==>...........................] - ETA: 9s - loss: 0.1096 - accuracy: 0.9657
#  7808/60000 [==>...........................] - ETA: 9s - loss: 0.1104 - accuracy: 0.9655
#  8192/60000 [===>..........................] - ETA: 9s - loss: 0.1106 - accuracy: 0.9652
#  8576/60000 [===>..........................] - ETA: 9s - loss: 0.1095 - accuracy: 0.9656
#  8960/60000 [===>..........................] - ETA: 9s - loss: 0.1098 - accuracy: 0.9655
#  9344/60000 [===>..........................] - ETA: 8s - loss: 0.1098 - accuracy: 0.9656
#  9728/60000 [===>..........................] - ETA: 8s - loss: 0.1100 - accuracy: 0.9655
# 10112/60000 [====>.........................] - ETA: 8s - loss: 0.1095 - accuracy: 0.9654
# 10496/60000 [====>.........................] - ETA: 8s - loss: 0.1085 - accuracy: 0.9660
# 10880/60000 [====>.........................] - ETA: 8s - loss: 0.1074 - accuracy: 0.9660
# 11264/60000 [====>.........................] - ETA: 8s - loss: 0.1081 - accuracy: 0.9656
# 11648/60000 [====>.........................] - ETA: 8s - loss: 0.1094 - accuracy: 0.9652
# 12032/60000 [=====>........................] - ETA: 8s - loss: 0.1093 - accuracy: 0.9652
# 12288/60000 [=====>........................] - ETA: 8s - loss: 0.1081 - accuracy: 0.9657
# 12672/60000 [=====>........................] - ETA: 8s - loss: 0.1066 - accuracy: 0.9662
# 12928/60000 [=====>........................] - ETA: 8s - loss: 0.1059 - accuracy: 0.9664
# 13184/60000 [=====>........................] - ETA: 8s - loss: 0.1051 - accuracy: 0.9667
# 13568/60000 [=====>........................] - ETA: 8s - loss: 0.1043 - accuracy: 0.9669
# 13952/60000 [=====>........................] - ETA: 8s - loss: 0.1042 - accuracy: 0.9670
# 14336/60000 [======>.......................] - ETA: 8s - loss: 0.1041 - accuracy: 0.9668
# 14592/60000 [======>.......................] - ETA: 8s - loss: 0.1045 - accuracy: 0.9667
# 14848/60000 [======>.......................] - ETA: 8s - loss: 0.1045 - accuracy: 0.9666
# 15232/60000 [======>.......................] - ETA: 8s - loss: 0.1054 - accuracy: 0.9666
# 15616/60000 [======>.......................] - ETA: 8s - loss: 0.1061 - accuracy: 0.9664
# 16000/60000 [=======>......................] - ETA: 7s - loss: 0.1057 - accuracy: 0.9666
# 16384/60000 [=======>......................] - ETA: 7s - loss: 0.1053 - accuracy: 0.9667
# 16768/60000 [=======>......................] - ETA: 7s - loss: 0.1052 - accuracy: 0.9667
# 17152/60000 [=======>......................] - ETA: 7s - loss: 0.1049 - accuracy: 0.9665
# 17536/60000 [=======>......................] - ETA: 7s - loss: 0.1056 - accuracy: 0.9662
# 17920/60000 [=======>......................] - ETA: 7s - loss: 0.1064 - accuracy: 0.9661
# 18304/60000 [========>.....................] - ETA: 7s - loss: 0.1066 - accuracy: 0.9661
# 18688/60000 [========>.....................] - ETA: 7s - loss: 0.1070 - accuracy: 0.9659
# 19072/60000 [========>.....................] - ETA: 7s - loss: 0.1068 - accuracy: 0.9659
# 19456/60000 [========>.....................] - ETA: 7s - loss: 0.1058 - accuracy: 0.9663
# 19840/60000 [========>.....................] - ETA: 7s - loss: 0.1048 - accuracy: 0.9667
# 20224/60000 [=========>....................] - ETA: 7s - loss: 0.1044 - accuracy: 0.9668
# 20608/60000 [=========>....................] - ETA: 7s - loss: 0.1042 - accuracy: 0.9668
# 20992/60000 [=========>....................] - ETA: 7s - loss: 0.1045 - accuracy: 0.9667
# 21248/60000 [=========>....................] - ETA: 7s - loss: 0.1040 - accuracy: 0.9669
# 21632/60000 [=========>....................] - ETA: 6s - loss: 0.1033 - accuracy: 0.9670
# 22016/60000 [==========>...................] - ETA: 6s - loss: 0.1030 - accuracy: 0.9670
# 22400/60000 [==========>...................] - ETA: 6s - loss: 0.1032 - accuracy: 0.9670
# 22784/60000 [==========>...................] - ETA: 6s - loss: 0.1029 - accuracy: 0.9672
# 23168/60000 [==========>...................] - ETA: 6s - loss: 0.1041 - accuracy: 0.9670
# 23552/60000 [==========>...................] - ETA: 6s - loss: 0.1039 - accuracy: 0.9672
# 23936/60000 [==========>...................] - ETA: 6s - loss: 0.1033 - accuracy: 0.9675
# 24192/60000 [===========>..................] - ETA: 6s - loss: 0.1030 - accuracy: 0.9675
# 24448/60000 [===========>..................] - ETA: 6s - loss: 0.1031 - accuracy: 0.9675
# 24832/60000 [===========>..................] - ETA: 6s - loss: 0.1027 - accuracy: 0.9676
# 25216/60000 [===========>..................] - ETA: 6s - loss: 0.1028 - accuracy: 0.9677
# 25600/60000 [===========>..................] - ETA: 6s - loss: 0.1027 - accuracy: 0.9678
# 25984/60000 [===========>..................] - ETA: 6s - loss: 0.1031 - accuracy: 0.9676
# 26368/60000 [============>.................] - ETA: 6s - loss: 0.1034 - accuracy: 0.9676
# 26624/60000 [============>.................] - ETA: 6s - loss: 0.1032 - accuracy: 0.9677
# 26880/60000 [============>.................] - ETA: 6s - loss: 0.1028 - accuracy: 0.9678
# 27136/60000 [============>.................] - ETA: 5s - loss: 0.1031 - accuracy: 0.9676
# 27392/60000 [============>.................] - ETA: 5s - loss: 0.1028 - accuracy: 0.9678
# 27776/60000 [============>.................] - ETA: 5s - loss: 0.1026 - accuracy: 0.9677
# 28032/60000 [=============>................] - ETA: 5s - loss: 0.1021 - accuracy: 0.9679
# 28416/60000 [=============>................] - ETA: 5s - loss: 0.1025 - accuracy: 0.9678
# 28800/60000 [=============>................] - ETA: 5s - loss: 0.1024 - accuracy: 0.9678
# 29184/60000 [=============>................] - ETA: 5s - loss: 0.1023 - accuracy: 0.9679
# 29568/60000 [=============>................] - ETA: 5s - loss: 0.1021 - accuracy: 0.9679
# 29952/60000 [=============>................] - ETA: 5s - loss: 0.1021 - accuracy: 0.9680
# 30336/60000 [==============>...............] - ETA: 5s - loss: 0.1025 - accuracy: 0.9679
# 30720/60000 [==============>...............] - ETA: 5s - loss: 0.1024 - accuracy: 0.9680
# 31104/60000 [==============>...............] - ETA: 5s - loss: 0.1021 - accuracy: 0.9681
# 31488/60000 [==============>...............] - ETA: 5s - loss: 0.1014 - accuracy: 0.9683
# 31872/60000 [==============>...............] - ETA: 5s - loss: 0.1009 - accuracy: 0.9685
# 32256/60000 [===============>..............] - ETA: 5s - loss: 0.1010 - accuracy: 0.9685
# 32640/60000 [===============>..............] - ETA: 4s - loss: 0.1012 - accuracy: 0.9684
# 33024/60000 [===============>..............] - ETA: 4s - loss: 0.1009 - accuracy: 0.9685
# 33408/60000 [===============>..............] - ETA: 4s - loss: 0.1009 - accuracy: 0.9686
# 33792/60000 [===============>..............] - ETA: 4s - loss: 0.1011 - accuracy: 0.9685
# 34176/60000 [================>.............] - ETA: 4s - loss: 0.1008 - accuracy: 0.9686
# 34560/60000 [================>.............] - ETA: 4s - loss: 0.1003 - accuracy: 0.9688
# 34944/60000 [================>.............] - ETA: 4s - loss: 0.1010 - accuracy: 0.9687
# 35328/60000 [================>.............] - ETA: 4s - loss: 0.1009 - accuracy: 0.9686
# 35712/60000 [================>.............] - ETA: 4s - loss: 0.1007 - accuracy: 0.9686
# 36096/60000 [=================>............] - ETA: 4s - loss: 0.1009 - accuracy: 0.9686
# 36480/60000 [=================>............] - ETA: 4s - loss: 0.1005 - accuracy: 0.9687
# 36864/60000 [=================>............] - ETA: 4s - loss: 0.1008 - accuracy: 0.9686
# 37248/60000 [=================>............] - ETA: 4s - loss: 0.1004 - accuracy: 0.9689
# 37632/60000 [=================>............] - ETA: 4s - loss: 0.1004 - accuracy: 0.9688
# 38016/60000 [==================>...........] - ETA: 3s - loss: 0.1006 - accuracy: 0.9689
# 38400/60000 [==================>...........] - ETA: 3s - loss: 0.1003 - accuracy: 0.9690
# 38784/60000 [==================>...........] - ETA: 3s - loss: 0.0999 - accuracy: 0.9691
# 39168/60000 [==================>...........] - ETA: 3s - loss: 0.0997 - accuracy: 0.9691
# 39552/60000 [==================>...........] - ETA: 3s - loss: 0.0998 - accuracy: 0.9692
# 39936/60000 [==================>...........] - ETA: 3s - loss: 0.0995 - accuracy: 0.9693
# 40320/60000 [===================>..........] - ETA: 3s - loss: 0.0994 - accuracy: 0.9694
# 40704/60000 [===================>..........] - ETA: 3s - loss: 0.0993 - accuracy: 0.9693
# 41088/60000 [===================>..........] - ETA: 3s - loss: 0.0989 - accuracy: 0.9694
# 41472/60000 [===================>..........] - ETA: 3s - loss: 0.0990 - accuracy: 0.9693
# 41856/60000 [===================>..........] - ETA: 3s - loss: 0.0989 - accuracy: 0.9694
# 42240/60000 [====================>.........] - ETA: 3s - loss: 0.0985 - accuracy: 0.9695
# 42624/60000 [====================>.........] - ETA: 3s - loss: 0.0983 - accuracy: 0.9696
# 43008/60000 [====================>.........] - ETA: 3s - loss: 0.0981 - accuracy: 0.9697
# 43264/60000 [====================>.........] - ETA: 3s - loss: 0.0980 - accuracy: 0.9697
# 43648/60000 [====================>.........] - ETA: 2s - loss: 0.0979 - accuracy: 0.9697
# 44032/60000 [=====================>........] - ETA: 2s - loss: 0.0975 - accuracy: 0.9697
# 44416/60000 [=====================>........] - ETA: 2s - loss: 0.0972 - accuracy: 0.9699
# 44800/60000 [=====================>........] - ETA: 2s - loss: 0.0972 - accuracy: 0.9698
# 45184/60000 [=====================>........] - ETA: 2s - loss: 0.0975 - accuracy: 0.9698
# 45568/60000 [=====================>........] - ETA: 2s - loss: 0.0976 - accuracy: 0.9698
# 45952/60000 [=====================>........] - ETA: 2s - loss: 0.0973 - accuracy: 0.9699
# 46336/60000 [======================>.......] - ETA: 2s - loss: 0.0973 - accuracy: 0.9699
# 46720/60000 [======================>.......] - ETA: 2s - loss: 0.0972 - accuracy: 0.9700
# 47104/60000 [======================>.......] - ETA: 2s - loss: 0.0971 - accuracy: 0.9700
# 47488/60000 [======================>.......] - ETA: 2s - loss: 0.0969 - accuracy: 0.9700
# 47872/60000 [======================>.......] - ETA: 2s - loss: 0.0969 - accuracy: 0.9699
# 48256/60000 [=======================>......] - ETA: 2s - loss: 0.0970 - accuracy: 0.9698
# 48640/60000 [=======================>......] - ETA: 2s - loss: 0.0968 - accuracy: 0.9699
# 49024/60000 [=======================>......] - ETA: 1s - loss: 0.0965 - accuracy: 0.9700
# 49408/60000 [=======================>......] - ETA: 1s - loss: 0.0964 - accuracy: 0.9699
# 49792/60000 [=======================>......] - ETA: 1s - loss: 0.0964 - accuracy: 0.9700
# 50176/60000 [========================>.....] - ETA: 1s - loss: 0.0962 - accuracy: 0.9700
# 50560/60000 [========================>.....] - ETA: 1s - loss: 0.0961 - accuracy: 0.9700
# 50944/60000 [========================>.....] - ETA: 1s - loss: 0.0960 - accuracy: 0.9701
# 51328/60000 [========================>.....] - ETA: 1s - loss: 0.0957 - accuracy: 0.9702
# 51712/60000 [========================>.....] - ETA: 1s - loss: 0.0956 - accuracy: 0.9701
# 52096/60000 [=========================>....] - ETA: 1s - loss: 0.0955 - accuracy: 0.9701
# 52480/60000 [=========================>....] - ETA: 1s - loss: 0.0955 - accuracy: 0.9701
# 52864/60000 [=========================>....] - ETA: 1s - loss: 0.0953 - accuracy: 0.9702
# 53248/60000 [=========================>....] - ETA: 1s - loss: 0.0955 - accuracy: 0.9702
# 53632/60000 [=========================>....] - ETA: 1s - loss: 0.0954 - accuracy: 0.9702
# 54016/60000 [==========================>...] - ETA: 1s - loss: 0.0952 - accuracy: 0.9702
# 54400/60000 [==========================>...] - ETA: 1s - loss: 0.0957 - accuracy: 0.9701
# 54784/60000 [==========================>...] - ETA: 0s - loss: 0.0957 - accuracy: 0.9701
# 55168/60000 [==========================>...] - ETA: 0s - loss: 0.0960 - accuracy: 0.9700
# 55552/60000 [==========================>...] - ETA: 0s - loss: 0.0961 - accuracy: 0.9700
# 55936/60000 [==========================>...] - ETA: 0s - loss: 0.0962 - accuracy: 0.9700
# 56320/60000 [===========================>..] - ETA: 0s - loss: 0.0960 - accuracy: 0.9700
# 56704/60000 [===========================>..] - ETA: 0s - loss: 0.0958 - accuracy: 0.9701
# 57088/60000 [===========================>..] - ETA: 0s - loss: 0.0958 - accuracy: 0.9701
# 57472/60000 [===========================>..] - ETA: 0s - loss: 0.0957 - accuracy: 0.9701
# 57856/60000 [===========================>..] - ETA: 0s - loss: 0.0959 - accuracy: 0.9701
# 58112/60000 [============================>.] - ETA: 0s - loss: 0.0960 - accuracy: 0.9700
# 58368/60000 [============================>.] - ETA: 0s - loss: 0.0958 - accuracy: 0.9701
# 58752/60000 [============================>.] - ETA: 0s - loss: 0.0956 - accuracy: 0.9702
# 59136/60000 [============================>.] - ETA: 0s - loss: 0.0955 - accuracy: 0.9702
# 59520/60000 [============================>.] - ETA: 0s - loss: 0.0953 - accuracy: 0.9703
# 59904/60000 [============================>.] - ETA: 0s - loss: 0.0952 - accuracy: 0.9703
# 60000/60000 [==============================] - 11s 184us/step - loss: 0.0952 - accuracy: 0.9703 - val_loss: 0.0708 - val_accuracy: 0.9769
#
#    32/10000 [..............................] - ETA: 0s
#   864/10000 [=>............................] - ETA: 0s
#  1696/10000 [====>.........................] - ETA: 0s
#  2528/10000 [======>.......................] - ETA: 0s
#  3424/10000 [=========>....................] - ETA: 0s
#  4224/10000 [===========>..................] - ETA: 0s
#  5024/10000 [==============>...............] - ETA: 0s
#  5856/10000 [================>.............] - ETA: 0s
#  6720/10000 [===================>..........] - ETA: 0s
#  7552/10000 [=====================>........] - ETA: 0s
#  8480/10000 [========================>.....] - ETA: 0s
#  9408/10000 [===========================>..] - ETA: 0s
# 10000/10000 [==============================] - 1s 59us/step
# 误差：0.07082823239136487591949276065861340612173
# 准确率： 0.9768999814987183