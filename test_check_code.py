from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import os
from random import choice



root_dir = "img/test"





img_list = []
def gen_list():

    for parent, dirnames, filenames in os.walk(root_dir):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for filename in filenames:  # 输出文件信息
            img_list.append(filename.replace(".gif",""))
            # print("parent is:" + parent)
            # print("filename is:" + filename)
            # print("the full name of the file is:" + os.path.join(parent, filename))  # 输出文件路径信息
    return img_list

img_list = gen_list()
def get_test_captcha_text_and_image(i=None):
    img = img_list[i]
    captcha_image = Image.open(root_dir + '/' + img + ".gif")
    captcha_image = np.array(captcha_image)
    return img, captcha_image