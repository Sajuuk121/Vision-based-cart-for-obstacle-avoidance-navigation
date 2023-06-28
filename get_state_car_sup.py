import numpy as np
import airsim
# import cv2 as cv
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
        
    def getImage(self):
        # get front camera scene
        response=self.client.simGetImages([airsim.ImageRequest("0",airsim.ImageType.Scene,False,False)])
        response=response[0]
        image_data=np.frombuffer(response.image_data_uint8,dtype=np.uint8)
        image_data=image_data.reshape(response.height,response.width,3)
        # print(image_data)
        # image_data=np.reshape(image_data,[response.height,response.width,3])
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
        '''
        terminal = False

        # ['forward','back','roll_right','roll_left','yaw_left','yaw_right','higher','lower']
        #tt=float(input_actions)
        #if tt <= 0.8:
            #self.carcontr.throttle=0.7
            #self.carcontr.steering = random.random() * random.randint(-1,1) * 0.6
            #self.carcontr.brake=0
        # elif tt <= 2:
        #     self.carcontr.throttle=0.3
        #     self.carcontr.steering = random.random() * random.randint(-1,1)
        #     self.carcontr.brake=0
        #else:
            #self.carcontr.brake=1
            #self.carcontr.throttle=0
            #self.carcontr.steering = 0
        #self.client.setCarControls(self.carcontr)
        #time.sleep(0.1)

        start_time = time.time()
        time_limit = 3

        # 在时间限制内，让车随机地自己走
        while time.time() - start_time < time_limit:
            # 生成随机的油门和转向值，范围在0到1之间
            throttle = random.random()
            steering = random.random() * random.choice([-1, 1])
            # 设置控制指令
            self.carcontr.throttle = throttle
            self.carcontr.steering = steering
            self.carcontr.brake = 0
            # 发送控制指令给车

            # 等待一小段时间，比如0.1秒

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