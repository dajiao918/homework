# -*- coding: UTF-8 -*-
import socket
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

face_detection_path = 'models/haarcascade_frontalface_default.xml'
fer_model_path = 'models/48.20_my_XCEPTION.42-0.69.hdf5'
# 载入人脸检测模型
face_detection = cv2.CascadeClassifier(face_detection_path)  # 级联分类器
# 载入人脸表情识别模型
emotion_classifier = load_model(fer_model_path, compile=False)
# 表情类别
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

s = socket.socket()         # 创建 socket 对象
host = '192.168.1.9'        # 获取本地主机名
port = 12345                # 设置端口
s.bind((host, port))        # 绑定端口
s.listen(5)                 # 等待客户端连接
print('开始监听端口：')

while True:
    c, addr = s.accept()     # 建立客户端连接
    print('收到{}请求'.format(addr))
    message = c.recv(1024)
    if message.decode() == '照片已生成':
        img = cv2.imread('../nao_Facial_expression_recognition/camImage.png')
        # cv2.imshow("Display window", img)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转为灰度图
        # cv2.imshow('Display window', gray)
        # cv2.waitKey(0)
        cv2.imwrite('photogray.png', gray)
        # 检测人脸
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                                minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        preds = []  # 预测的结果
        label = None  # 预测的标签
        (fX, fY, fW, fH) = None, None, None, None  # 人脸位置
        if len(faces) > 0:
            # 选择检测到的ROI最大的人脸
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces

            # 从灰度图中提取感兴趣区域（ROI），将其大小转换为48*48 pixels，并为通过CNN的分类器准备ROI
            roi = gray[fY - 5:fY + fH + 5, fX - 5:fX + fW + 5]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # 用模型预测各分类的概率
            preds = emotion_classifier.predict(roi)[0]
            # emotion_probability = np.max(preds)  # 最大的概率
            label = EMOTIONS[preds.argmax()]  # 选取最大概率的表情类
            print(label)
            c.send(bytes(label, 'utf-8'))
        else:
            c.send(b'no face')
