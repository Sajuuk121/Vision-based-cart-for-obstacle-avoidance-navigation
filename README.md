# Vision-based-cart-for-obstacle-avoidance-navigation
# 基于视觉的避障小车
## abstract
This is  repository to introduce Vision-based cart for obstacle avoidance navigation, using Airsim and Tensorflow.
This report introduces the use of the AirSim simulation platform and the implementation of visual obstacle avoidance based on deep learning.AirSim is an open source cross-platform simulator based on a game engine, which can be used for physical and visual simulation of robots such as UAVs and unmanned vehicles. It supports both software-in-the-loop simulation based on flight controllers such as PX4 and ArduPilot, and currently supports hardware-in-the-loop simulation based on PX4. AirSim features very good visual simulation and is very suitable for performing visual AI simulation verification based on deep learning, etc. This report introduces how to use the API interface of AirSim to connect and control the simulation cart and obtain the state information of the cart, using the BLOCKS scenario as an example. This report also introduces the basic concepts and functions of the TensorFlow framework, and how to use the RESNET network to classify the images received by the optical camera on the cart to achieve the function of visual obstacle avoidance. This report concludes with a summary of the strengths and weaknesses of this solution, and proposes the next work plan.

## 摘 要
本报告主要介绍了AirSim仿真平台的使用方法和基于深度学习的视觉避障的实现方法。AirSim是一款基于游戏引擎的开源跨平台仿真器，它可以用于无人机、无人车等机器人的物理和视觉仿真。它同时支持基于PX4和ArduPilot等飞行控制器的软件在环仿真，目前还支持基于PX4的硬件在环仿真。AirSim的特点是具有非常优秀的视觉仿真效果，非常适用于进行基于深度学习等视觉AI仿真验证。本报告以BLOCKS场景为例，介绍了如何利用AirSim的API接口来连接和控制仿真小车，并获取小车的状态信息。本报告还介绍了TensorFlow框架的基本概念和功能，以及如何利用RESNET网络来对小车上的光学摄像头接收到的图像进行分类，从而实现视觉避障的功能。本报告最后总结了本次方案的优点和不足，并提出了下一步工作计划。

关键词：**视觉导航**；**AirSim**；**深度学习**；**视觉避障**

## 目    录    


第一章  绪论	 

1.1  研究背景   

1.2  视觉导航的方法与发展趋势   


1.2.1视觉导航的方法	   

1.2.2视觉导航的发展趋势	    

1.2.3视觉导航的应用案例  

第二章  AirSim仿真平台介绍与连接	

2.1  AirSim平台  

2.2 代码与连接	 

第三章  基于深度学习的视觉避障   

3.1  Tensorflow的介绍	

3.1.1  训练网络搭建  

3.1.2  训练主程序	

第四章  总结	

4.1  方案总结	

4.1.1  方案总结	

4.1.2  存在问题	  

4.2  下一步工作	

## 第一章  绪论
**1.1  研究背景**   

视觉导航（Visual Navigation）是指让机器人或智能车辆能够根据视觉信息（如摄像头拍摄的图像）来感知周围环境，定位自身位置，规划合适的路径，并执行相应的动作，从而在未知或部分已知的环境中实现目标导航的能力。   

视觉导航是人工智能领域，机器人领域非常重要的一个研究方向，它涉及到计算机视觉、机器学习、强化学习、控制理论等多个学科的交叉与融合。视觉导航具有广泛的应用场景，如无人驾驶、无人机、服务机器人、探索机器人等。    

视觉导航是一个非常复杂和困难的问题，它面临着以下几个主要的挑战：    

1）视觉感知的不准确性和不稳定性。由于环境中存在各种光照、遮挡、动态变化、噪声等干扰因素，视觉感知往往难以获得完整、清晰、准确的信息，从而影响后续的定位和规划。 

2）环境的不确定性和复杂性。环境中可能存在各种未知或难以预测的情况，如突然出现的障碍物、行人或车辆，或者路况发生变化等，这些都要求视觉导航系统具有较强的适应性和鲁棒性。 

3）决策和控制的高效性和安全性。视觉导航系统需要在有限的时间和资源内做出合理的决策，并执行相应的控制命令，以达到目标位置。同时，还要保证在行驶过程中避免碰撞或危险情况，确保安全性。     


**1.2  视觉导航的方法与发展趋势**   

1.2.1视觉导航的方法 

为了解决视觉导航问题，研究者们提出了各种各样的方法，大致可以分为以下几类：  

1）基于地图的方法。这类方法需要事先建立或提供一个环境地图（如拓扑地图、栅格地图、语义地图等），然后利用视觉信息来定位自身在地图中的位置，并根据地图信息来规划路径和执行动作。这类方法通常需要较大的存储空间和计算资源，且对地图质量和更新有较高的要求。 

2）基于标记点的方法。这类方法不需要完整的环境地图，而是利用一些预先设置或自动提取的标记点（如二维码、路标、特征点等）来辅助定位和规划。这类方法可以减少存储和计算开销，但对标记点的数量和分布有较高的依赖性，且难以处理动态环境。   

3）基于学习的方法。这类方法不需要事先提供任何地图或标记点信息，而是利用机器学习或深度学习的技术，让视觉导航系统从大量的数据中自主学习如何感知、定位、规划和控制。这类方法可以适应各种未知或复杂的环境，且具有较强的泛化能力，但需要大量的训练数据和计算资源，且难以保证安全性和可解释性。   


| 标题 | 作者 | 链接 |
| ----------- | ----------- | ------ |
| Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments      | Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, Anton van den Hengel     | arxiv.org/abs/1711.07280 | 
| Cognitive Mapping and Planning for Visual Navigation      | Saurabh Gupta, Varun Tolani, James Davidson, Sergey Levine, Rahul Sukthankar, Jitendra Malik     | arxiv.org/abs/1702.03920 | 

从这些论文中，我们可以看出以下几个视觉导航的发展趋势：  

1）端到端的神经网络模型。这些模型可以直接从原始的图像输入到动作输出，无需进行复杂的中间处理，简化了系统的设计和实现，提高了系统的效率和性能。   

2）基于强化学习或模仿学习的训练方法。这些方法可以让视觉导航系统从大量的数据中自动学习最优或次优的策略，无需人为地设定规则或目标函数，增强了系统的适应性和鲁棒性。   

3）基于纯视觉或多模态的感知方式。这些方式可以利用摄像头或其他传感器（如雷达、激光等）来获取更丰富和更准确的环境信息，无需依赖于地图或标记点，扩展了系统的应用范围和场景 

1.2.3视觉导航的应用案例
视觉导航已经在许多领域和场景中得到了广泛的应用和验证，以下是一些典型的应用案例：    

1）特斯拉无人驾驶。特斯拉是全球领先的电动汽车和智能汽车的领导者，其无人驾驶系统主要依赖于视觉导航技术，利用多个摄像头、雷达和超声波传感器来感知周围的车辆、行人、交通标志和信号灯等，然后通过深度神经网络来分析和处理视觉信息，生成驾驶决策和控制命令。特斯拉的视觉导航系统具有以下特点：   

- 不依赖于高精度地图和激光雷达，而是通过纯视觉和人工智能来实现自动驾驶的能力。  

- 可以在不同的道路类型、天气条件、光照环境和交通场景下进行视觉导航，具有较强的泛化能力和鲁棒性。    

- 可以通过云端的数据收集和模型更新，不断提升视觉导航的性能和安全性。    

特斯拉的视觉导航系统已经在全球范围内部署了数百万辆汽车，为用户提供了便捷、舒适和安全的出行体验。特斯拉的视觉导航系统也在不断进化和完善，目前已经发布了10.1版本的全自动驾驶软件（Full Self-Driving，FSD），可以实现更多的自动驾驶功能，如自动变道、自动停车、自动调头等。    


2）无人机视觉导航。无人机是一种可以在空中自主或遥控飞行的机器人，它具有广泛的应用场景，如军事侦察、灾难救援、航拍摄影等。无人机视觉导航是指让无人机能够根据机载摄像头拍摄的图像来感知周围环境，定位自身位置，规划合适的路径，并执行相应的动作，从而在未知或部分已知的环境中实现目标导航的能力。 

无人机视觉导航面临着以下几个主要的挑战：    

1）视觉感知的不准确性和不稳定性。由于空中环境中存在各种光照、遮挡、动态变化、噪声等干扰因素，视觉感知往往难以获得完整、清晰、准确的信息，从而影响后续的定位和规划。 

2）环境的不确定性和复杂性。环境中可能存在各种未知或难以预测的情况，如突然出现的障碍物、飞鸟或飞机，或者风速风向发生变化等，这些都要求无人机视觉导航系统具有较强的适应性和鲁棒性。   

3）决策和控制的高效性和安全性。无人机视觉导航系统需要在有限的时间和资源内做出合理的决策，并执行相应的控制命令，以达到目标位置。同时，还要保证在飞行过程中避免碰撞或危险情况，确保安全性。
为了解决无人机视觉导航问题，研究者们提出了各种各样的方法，大致可以分为以下几类：    

- 基于地图的方法。这类方法需要事先建立或提供一个环境地图（如拓扑地图、栅格地图、语义地图等），然后利用视觉信息来定位自身在地图中的位置，并根据地图信息来规划路径和执行动作。这类方法通常需要较大的存储空间和计算资源，且对地图质量和更新有较高的要求。  

- 基于标记点的方法。这类方法不需要完整的环境地图，而是利用一些预先设置或自动提取的标记点（如二维码、路标、特征点等）来辅助定位和规划。这类方法可以减少存储和计算开销，但对标记点的数量和分布有较高的依赖性，且难以处理动态环境。    

- 基于学习的方法。这类方法不需要事先提供任何地图或标记点信息，而是利用机器学习或深度学习的技术，让无人机视觉导航系统从大量的数据中自主学习如何感知、定位、规划和控制。这类方法可以适应各种未知或复杂的环境，且具有较强的泛化能力，但需要大量的训练数据和计算资源，且难以保证安全性和可解释性。  

无人机视觉导航已经在许多领域和场景中得到了广泛的应用和验证，以下是一些典型的应用案例：  

1）Skydio 2无人机。Skydio 2是一款美国公司Skydio开发的智能无人机，它主要依赖于视觉导航技术，利用6个4K摄像头、NVIDIA Jetson TX2芯片和深度神经网络来感知周围的环境，实现了全自动避障功能。Skydio 2无人机可以在各种复杂的环境中自主飞行，如森林、城市、山地等，可以跟随用户或指定目标进行航拍摄影。 

2）大疆无人机。大疆是全球领先的无人机制造商，其无人机主要依赖于视觉导航技术，利用多个视觉传感器、红外传感器、超声波传感器、GPS/GLONASS双模卫星定位系统等来感知周围的环境，实现了多种自动驾驶功能，如自动起飞降落、自动返航、自动避障、自动跟随等。大疆无人机可以在不同的道路类型、天气条件、光照环境和交通场景下进行视觉导航，具有较强的适应性和鲁棒性。    

## 第二章  AirSim仿真平台介绍与连接
**2.1  AirSim平台**
AirSim是一款基于游戏引擎的开源跨平台仿真器，它可以用于无人机、无人车等机器人的物理和视觉仿真。它同时支持基于PX4和ArduPilot等飞行控制器的软件在环仿真，目前还支持基于PX4的硬件在环仿真
AirSim是微软公司最早于2017年开始开发的，经过6年的迭代更新，现已经发展为功能完善的以视觉仿真为特点的无人机和无人车仿真平台。 

AirSim是作为虚幻游戏引擎的插件而存在的，适配基于虚幻引擎搭建的环境。目前AirSim也有一个实验版本的插件适用于Unity引擎。   

得益于游戏引擎优秀的视觉渲染效果，AirSim仿真平台的一大亮点是具有非常优秀的视觉仿真效果，非常适用于进行基于深度学习等视觉AI仿真验证。    

AirSim开放了很多API接口，用于读取数据、控制车辆、控制天气等，AirSim的开发者希望这个平台能够帮助研究人员用于深度学习、计算机视觉、强化学习等人工智能相关的研究。 

具体的地址可以参考：https://microsoft.github.io/AirSim/apis/

**2.2 代码与连接**
2.2.1 AirSim例程    

首先是利用API的接口，以Python的形式来进行连接，控制仿真小车。   

    import airsim
    import time

    # 先打开仿真平台，然后进行连接
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    while True:
     # 得到小车的状态
     car_state = client.getCarState()
        print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

        #设定小车控制律
        car_controls.throttle = 1  #节流阀
        car_controls.steering = 1  #转弯
        client.setCarControls(car_controls)  

        # 让小车动一动
        time.sleep(1)

        # 得到小车上的模拟摄像头照片
        responses = client.simGetImages([
            airsim.ImageRequest(0, airsim.ImageType.DepthVis),
            airsim.ImageRequest(1, airsim.ImageType.DepthPlanar, True)])
        print('Retrieved images: %d', len(responses))

        # 进行处理
        for response in responses:
          if response.pixels_as_float:
                print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
             airsim.write_pfm('py1.pfm', airsim.get_pfm_array(response))
          else:
              print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
              airsim.write_file('py1.png', response.image_data_uint8)

2.2.2得到模拟小车的状态 

通过例程后就可以连接小车，并且得知小车状态。

    import airsim
    import time
    import random

    '环境交互类 获取airsim飞行状态,用于监督学习的状态，并且实现控制车'
    class FlyingState:
       def __init__(self):
         # self.dest=(x,y,z)
         self.client=airsim.CarClient()
            self.linkToAirsim()
            # self.client.moveToPositionAsync()自动移动到目的地
            # client take off


        def linkToAirsim(self):
         # 连接到airsim
          self.client.confirmConnection()
         self.client.reset()
          self.client.enableApiControl(True)
          self.client.armDisarm(True)
          self.carcontr=self.client.getCarControls()
         # connect to the AirSim simulator 
         return self.getImage()
        
    #同上
     def getImage(self):
         # 得到前视摄像头的照片
         response=self.client.simGetImages([airsim.ImageRequest("0",airsim.ImageType.Scene,False,False)])
         response=response[0]
         image_data=np.frombuffer(response.image_data_uint8,dtype=np.uint8)
         image_data=image_data.reshape(response.height,response.width,3)
     
    image_data=np.reshape(image_data,[response.height,response.width,3])
         image_data=np.flipud(image_data)
         # image_data=image_data[:,:]
         # # [height,width,channel]=image_data.shape
         # _, image_data = cv.threshold(image_data, 0, 255, cv.THRESH_BINARY)
         return image_data


     def frame_step(self, input_actions):
         # 执行操作并获取帧
         '''
         input_actions 包含三个参数: 
         [0]throttle [1]break [2]steering
         控制的算法
    0是油门/节流阀、1是刹车、2是转弯
            '''
         terminal = False

         # ['forward','back','roll_right','roll_left','yaw_left','yaw_right','higher','lower']
          tt=float(input_actions)
          if tt <= 0.8:
             self.carcontr.throttle=0.7
             self.carcontr.steering = random.random() * random.randint(-1,1) * 0.6
             self.carcontr.brake=0
         # elif tt <= 2:
         #     self.carcontr.throttle=0.3
         #     self.carcontr.steering = random.random() * random.randint(-1,1)
         #     self.carcontr.brake=0
         else:
             self.carcontr.brake=1
             self.carcontr.throttle=0
             self.carcontr.steering = 0
         self.client.setCarControls(self.carcontr)
         time.sleep(0.1)
         # client state
         # client_state=self.client.getCarState().kinematics_estimated
         # client_vel = client_state.linear_velocity
         # position and crash
         Crash_info = self.client.simGetCollisionInfo().has_collided

         reward = 0
         if Crash_info :
             terminal = True
             reward = 1

            return self.getImage(), reward, terminal

        def rand_action(self):
         for i in range(5):
             self.carcontr.throttle=1
             self.client.setCarControls(self.carcontr)





    # responses=client.simGetImages([
    #     airsim.ImageRequest(0,airsim.ImageType.Scene),
    #     # 前视深度信息
    #     airsim.ImageRequest(0,airsim.ImageType.DepthVis),
    #     # bottom深度信息
    #     airsim.ImageRequest(3,airsim.ImageType.DepthVis)
    # ])
    if __name__ == "__main__":
     '测试飞行'
     game_state = FlyingState()
     for i in range(0,1):
         game_state.rand_action()

现在已经得到了小车的状态，有了小车的状态后就可以开始利用深度学习的网络来进行学习，达到视觉避障的目的。

## 第三章  基于深度学习的视觉避障
**3.1  Tensorflow的介绍**   

TensorFlow是一个端到端的开源机器学习平台，它提供了灵活的工具、库和社区资源，让任何人都可以轻松地构建和部署机器学习应用。    

TensorFlow的核心是一个用于表示计算的数据流图，其中每个节点代表一个数学运算，每条边代表一个多维数组（张量）。这种表示方式可以让你在不同的硬件平台上高效地执行复杂的计算，包括CPU、GPU、TPU等。   

TensorFlow支持多种编程语言，比如Python、C++、Java等，但是最常用的是Python。TensorFlow提供了多个高级API，比如Keras、Estimator等，让你可以用简洁的代码构建和训练神经网络。    

TensorFlow还有一个丰富的生态系统，包括各种工具和库，比如：  

1）TensorFlow Hub：一个提供预训练模型和数据集的平台，让你可以重用别人的经验和知识。 

2）TensorFlow Lite：一个轻量级的框架，让你可以在移动设备或嵌入式设备上运行机器学习模型。    

3）TensorFlow.js：一个JavaScript库，让你可以在浏览器或Node.js上运行机器学习模型。   

4）TensorFlow Extended：一个端到端的平台，让你可以在生产环境中部署和管理机器学习管道。  

其他可参考官方网站：    

 [tensorflow.google.cn](https://tensorflow.google.cn/)


**3.1.1  训练网络搭建** 

本质上我们使用到的深度学习是用来对图片进行分类，有两大类，一大类是有障碍物遮挡、一大类是没有障碍物遮挡。然后利用网络进行判断，如果摄像头里面拍到的图像是有遮挡，那么改变当前轨迹例如刹车和转弯。    

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
          # linerchange
          self.l_1 = Conv2D(filters=128, kernel_size=1, padding='same',strides=2,
                              kernel_initializer ='he_uniform',)  # 卷积层)
          # linerchange
          self.l_2 = Conv2D(filters=256, kernel_size=1, padding='same',strides=2,
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

          self.p_2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')  # 池化层
          self.flatten = Flatten()
          self.f1 = Dense(256, activation='elu',
                            kernel_initializer='he_uniform')
          self.f1_2 = Dense(32, activation='elu',
                            kernel_initializer='he_uniform')
         self.f2 = Dense(1, activation=None,
                             kernel_initializer='he_uniform') # 输出层
          # self.f3 = Dense(1, activation='tanh',
          #                    kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=42),
                          #    bias_initializer = 'he_uniform')
          # 加载网络
         self.checkpoint_save_path = "./model_sup/car_actor1"
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
           x1 = self.l_1(x1)
           x = self.c_4_1(x1)
           x = self.b_4_1(x)
           x = self.a_4_1(x)
           x = self.c_4_2(x)
           x = self.b_4_2(x+x1)
         x1 = self.a_4_2(x)
            x1 = self.l_2(x1)
            x = self.p_2(x1)
           x = self.flatten(x)
          x = self.f1(x)
          x = self.f1_2(x)
           y = self.f2(x) # 要考虑输入输出的维度
           y = tf.squeeze(y,1)
          return y      
       def save_wei(self):
          # 保存网络
           self.save(self.checkpoint_save_path)


**3.1.2  训练主程序** 

主程序就利用前面已经构建好的网络模型和连接AirSim的程序，来进行模型的训练。

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

## 第四章  总结
**4.1  方案总结**   

**4.1.1  方案总结** 

在本次实验中，我选择下载了BLOCKS的场景来进行训练，在该场景中，有比较简单的障碍物。
我们主要通过利用深度学习中的RESNET网络来对车上的光学摄像头接收到的信息进行图像的分类，依据分类的结果对小车采取不同的策略。例如如果前视摄像头中发现了有障碍物即block遮挡时，小车就以0.323的油门和-0.774的转弯进行障碍物避障，这是由于在原始的BLOCKS中，以该转弯率和油门则会原地打转，不会撞上任何障碍物。如果没有则直行。    


**4.1.2  存在问题**

最主要的问题是算法的鲁棒性不够好，拓展性不够好，在小车上进行算法实测的时候发现小车对于教室场景的识别不佳。推测问题在于场景的真实性不够，所以我加入了真实的数据集来进行训练，以小车真实自带的摄像头拍摄的图片来进行训练，加入该数据集后的训练训练了50轮，得到的结果与之前相比有一定的提高。        

**4.2  下一步工作** 

可以利用小车收集更多的数据集来进行训练。










