import tensorflow as tf
from keras import Model
from keras.layers import   Dense, Activation, Conv2D, MaxPool2D, AveragePooling2D, Flatten, BatchNormalization, LayerNormalization,AveragePooling1D
import numpy as np
import os 
'简单的resnet模型, 用于监督学习'
class Actor(Model): 
    # 评估网络,输出动作
    def __init__(self):
        super().__init__() 
        # resnet
        self.c_1_1 = Conv2D(filters=64, kernel_size=(5, 5),strides=2, padding='valid', 
                           kernel_initializer='he_uniform',)  # 卷积层
        self.b_1 = BatchNormalization()
        self.a_1_1 = Activation('elu')  # 激活层
        self.p_1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')  # 池化层
        # c2
        self.c_2_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                           kernel_initializer ='he_uniform',)  # 卷积层
        self.b_2_1 = BatchNormalization()
        self.a_2_1 = Activation('elu')  # 激活层
        self.c_2_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                           kernel_initializer='he_uniform',)  # 卷积层
        self.b_2_2 = BatchNormalization()
        self.a_2_2 = Activation('elu')  # 激活层
        # c3
        self.c_3_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                           kernel_initializer='he_uniform',)  # 卷积层
        self.b_3_1 = BatchNormalization()
        self.a_3_1 = Activation('elu')  # 激活层
        self.c_3_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                           kernel_initializer ='he_uniform',)  # 卷积层
        self.b_3_2 = BatchNormalization()
        self.a_3_2 = Activation('elu')  # 激活层
        # linerchange
        self.l_1 = Conv2D(filters=128, kernel_size=1, padding='same',strides=2,
                           kernel_initializer ='he_uniform',)  # 卷积层)
        # c4
        self.c_4_1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', 
                           kernel_initializer='he_uniform',)  # 卷积层
        self.b_4_1 = BatchNormalization()
        self.a_4_1 = Activation('elu')  # 激活层
        self.c_4_2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                           kernel_initializer ='he_uniform',)  # 卷积层
        self.b_4_2 = BatchNormalization()
        self.a_4_2 = Activation('elu')  # 激活层
        # c5
        self.c_5_1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                           kernel_initializer='he_uniform',)  # 卷积层
        self.b_5_1 = BatchNormalization()
        self.a_5_1 = Activation('elu')  # 激活层
        self.c_5_2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                           kernel_initializer ='he_uniform',)  # 卷积层
        self.b_5_2 = BatchNormalization()
        self.a_5_2 = Activation('elu')  # 激活层
        # linerchange
        self.l_2 = Conv2D(filters=256, kernel_size=1, padding='same',strides=2,
                           kernel_initializer ='he_uniform',)  # 卷积层)
        # c6
        self.c_6_1 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', 
                           kernel_initializer='he_uniform',)  # 卷积层
        self.b_6_1 = BatchNormalization()
        self.a_6_1 = Activation('elu')  # 激活层
        self.c_6_2 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                           kernel_initializer ='he_uniform',)  # 卷积层
        self.b_6_2 = BatchNormalization()
        self.a_6_2 = Activation('elu')  # 激活层
        # c7
        self.c_7_1 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                           kernel_initializer='he_uniform',)  # 卷积层
        self.b_7_1 = BatchNormalization()
        self.a_7_1 = Activation('elu')  # 激活层
        self.c_7_2 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                           kernel_initializer ='he_uniform',)  # 卷积层
        self.b_7_2 = BatchNormalization()
        self.a_7_2 = Activation('elu')  # 激活层
        # linerchange
        self.l_3 = Conv2D(filters=512, kernel_size=1, padding='same',strides=2,
                           kernel_initializer ='he_uniform',)  # 卷积层)
        # c8
        self.c_8_1 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', 
                           kernel_initializer='he_uniform',)  # 卷积层
        self.b_8_1 = BatchNormalization()
        self.a_8_1 = Activation('elu')  # 激活层
        self.c_8_2 = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                           kernel_initializer ='he_uniform',)  # 卷积层
        self.b_8_2 = BatchNormalization()
        self.a_8_2 = Activation('elu')  # 激活层
        # c9
        self.c_9_1 = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                           kernel_initializer='he_uniform',)  # 卷积层
        self.b_9_1 = BatchNormalization()
        self.a_9_1 = Activation('elu')  # 激活层
        self.c_9_2 = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                           kernel_initializer ='he_uniform',)  # 卷积层
        self.b_9_2 = BatchNormalization()
        self.a_9_2 = Activation('elu')  # 激活层
        
        self.p_2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')  # 池化层
        self.flatten = Flatten()
        self.f1 = Dense(256, activation='elu',
                           kernel_initializer='he_uniform',)
        self.f1_2 = Dense(32, activation='elu',
                           kernel_initializer='he_uniform',)
        self.f2 = Dense(1, activation='softmax',
                           kernel_initializer='he_uniform',) # 输出层
        # self.f3 = Dense(1, activation='tanh',
        #                    kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=42),
                        #    bias_initializer = 'he_uniform')
        # 加载网络
        self.checkpoint_save_path = "./model_sup/car_actor"
        if os.path.exists(self.checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            self.load_weights(self.checkpoint_save_path)
        else:
            print('-------------train new model-----------------')

    def call(self,x):
        x = self.c_1_1(x)
        x = self.b_1(x)
        x = self.a_1_1(x)
        x1 = self.p_1(x)
        x = self.c_2_1(x1)
        x = self.b_2_1(x)
        x = self.a_2_1(x)
        x = self.c_2_2(x)
        x = self.b_2_2(x+x1)
        x1 = self.a_2_2(x)
        x = self.c_3_1(x1)
        x = self.b_3_1(x)
        x = self.a_3_1(x)
        x = self.c_3_2(x)
        x = self.b_3_2(x+x1)
        x1 = self.a_3_2(x)
        x1 = self.l_1(x1)
        x = self.c_4_1(x1)
        x = self.b_4_1(x)
        x = self.a_4_1(x)
        x = self.c_4_2(x)
        x = self.b_4_2(x+x1)
        x1 = self.a_4_2(x)
        x = self.c_5_1(x1)
        x = self.b_5_1(x)
        x = self.a_5_1(x)
        x = self.c_5_2(x)
        x = self.b_5_2(x+x1)
        x1 = self.a_5_2(x)
        x1 = self.l_2(x1)
        x = self.c_6_1(x1)
        x = self.b_6_1(x)
        x = self.a_6_1(x)
        x = self.c_6_2(x)
        x = self.b_6_2(x+x1)
        x1 = self.a_6_2(x)
        # x = self.c_7_1(x1)
        # x = self.b_7_1(x)
        # x = self.a_7_1(x)
        # x = self.c_7_2(x)
        # x = self.b_7_2(x+x1)
        # x1 = self.a_7_2(x)
        # x1 = self.l_3(x1)
        # x = self.c_8_1(x1)
        # x = self.b_8_1(x)
        # x = self.a_8_1(x)
        # x = self.c_8_2(x)
        # x = self.b_8_2(x+x1)
        # x1 = self.a_8_2(x)
        # x = self.c_9_1(x1)
        # x = self.b_9_1(x)
        # x = self.a_9_1(x)
        # x = self.c_9_2(x)
        # x = self.b_9_2(x+x1)
        # x1 = self.a_9_2(x)
        x = self.p_2(x1)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.f1_2(x)
        y = self.f2(x) # 要考虑输入输出的维度
        # y2 = self.f3(x)
        # y = tf.concat([y1,y2],1)
        y = tf.reshape(y,[-1])
        return y      
    def save_wei(self):
        # 保存网络
        self.save_weights(self.checkpoint_save_path)
