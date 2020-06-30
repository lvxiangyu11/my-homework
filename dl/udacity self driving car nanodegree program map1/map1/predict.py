import base64
import json
import numpy as np

import socketio
import eventlet
import eventlet.wsgi
import time

from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template  # Flask是一个网络框架，方便写网络程序
from io import BytesIO

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import cv2

sio = socketio.Server()  # 服务器
app = Flask(__name__)  # 创建网络框架
model = None

# socket接收到数据，调用telemetry方法
@sio.on('telemetry')
def telemetry(sid, data):
    # 当前车的方向盘转动角
    steering_angle = data["steering_angle"]
    # 当前油门
    throttle = data["throttle"]
    # 当前车速
    speed = data["speed"]
    # 车中间位置的摄像头捕捉的画面
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    # BGR->RGB
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # 缩放图像到网络输入要求的大小
    image_array = image_array[80:140, 0: 320]
    # 正规化图像
    image_array = cv2.resize(image_array, (128, 128)) / 255. - 0.5
    # 图像从3维增加一个批处理维度
    transformed_image_array = image_array[None, :, :, :]

    # 预测角度
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    # 设置油门常数为1
    throttle = 1

    # 输出预测的角度和油门到命令行
    print(steering_angle, throttle)

    # 发送方向盘转动角和油门给模拟器
    send_control(steering_angle, throttle)


# 建立链接
@sio.on('connect')
def connect(sid, environ):
    print("connecting finished!", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    model = load_model('model.h5')

    # 将Flask应用绑定到中间件上去
    app = socketio.Middleware(sio, app)

    # 启动eventlet WSGI 服务器， 监听4567端口
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
