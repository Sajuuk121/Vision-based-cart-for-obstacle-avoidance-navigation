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


| 标题 | 作者 | 链接 | 代码 |
| ----------- | ----------- | ------ | ---- |
| Learning to Navigate in Cities Without a Map      | Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, Anton van den Hengel     | arxiv.org/abs/1711.07280 | github.com/YicongHong/Egithub.com/peteanderson |

