import tensorflow as tf
from keras import losses
import keras.optimizers as optimizers
import random
import numpy as np
from collections import deque
import os 
import datetime
from imgnet_sup import Actor
from get_state_car_sup import FlyingState

os.environ['CUDA_VISIBLE_DEVICES']='0'
'训练算法文件'

# 常量
REPLAY_MEMORY = 3200 # 观测存储器D的容量
BATCH = 64 # 训练batch大小
OBSERVE = BATCH+5 # 训练前观察积累的轮数

def trainNet(istrain):
    # 创建网络
    actor_val=Actor()
    # tf.random.set_seed(42)

    # 将每一轮的观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
    D = deque()  # Memory
    t=0
    temp_t=0
    s = env.linkToAirsim() # reset
    s = tf.convert_to_tensor(s,tf.float32)
    optimizer_ac = tf.keras.optimizers.Adam(learning_rate = 1e-5)

    eps=0
    while eps < 1501:
        action = actor_val(tf.expand_dims(s, 0)) 
        s_t,r,d = env.frame_step(action)
        s_t = tf.convert_to_tensor(s_t,tf.float32)
        temp_t+=1
        D.append((s,r))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        s=s_t
        t+=1


#============================ 训练网络 ===========================================
        # 观测一定轮数后开始训练
        if  t > OBSERVE and istrain and temp_t > 20:
            # 随机抽取minibatch个数据训练
            eps+=1
            for i in range(60):
                # print("==================start train====================t=",t)

                minibatch = random.sample(D, BATCH)

                # 获得batch中的每一个变量
                b_s = tf.convert_to_tensor([d[0] for d in minibatch])
                b_r = tf.convert_to_tensor([d[1] for d in minibatch],dtype=tf.float32)

                # 训练Critic
                with tf.GradientTape() as tape:
                    loss1 = losses.MSE(b_r,actor_val(b_s))                
                    # print("loss1 = %f " % loss1)
                gradients = tape.gradient(loss1, actor_val.trainable_variables)
                optimizer_ac.apply_gradients(zip(gradients, actor_val.trainable_variables))

                if i%4 == 3:
                # tensorboard
                    print("ep=",eps,"loss1 = %f " % loss1)

            temp_t=0

        if d == True:
            
            s=env.linkToAirsim() # 完成后reset
            s = tf.convert_to_tensor(s,tf.float32)
            ep_reward=0

def test():
    t=0
    actor_val=Actor('val')
    s = env.linkToAirsim()
    ep_reward = 0
    while t < 2000:

        action=actor_val(tf.expand_dims(tf.constant(s, dtype=tf.float32), 0)) # 注意单个动作要expand dim
        s_t,r,d= env.frame_step(action.numpy()[0])
        ep_reward+=r
        s=s_t
        t+=1

        if d==True:
            break

if __name__ == "__main__":
    env = FlyingState()
    trainNet(True)

