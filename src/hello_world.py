
import cv2  # 导入opencv库
import numpy as np  # 导入numpy库
import insightface  # 导入insightface库

from insightface.app import FaceAnalysis  # 导入FaceAnalysis类
from insightface.data import get_image as ins_get_image  # 导入get_image函数

# 创建FaceAnalysis类对象
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

app.prepare(ctx_id=0, det_size=(640, 640))  # 准备模型

img = ins_get_image('t1')  # 获取图片

faces = app.get(img)  # 获取人脸

rimg = app.draw_on(img, faces)  # 在图片上画人脸

cv2.imwrite("./t1_output.jpg", rimg)  # 保存图片
