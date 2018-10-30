from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random
# import matplotlib.pyplot as plt
import os
from random import choice

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#             'v', 'w', 'x', 'y', 'z']
# ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#             'V', 'W', 'X', 'Y', 'Z']

root_dir = "img/train"

# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image



def gen_list():
    img_list = []
    for parent, dirnames, filenames in os.walk(root_dir):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for filename in filenames:  # 输出文件信息
            img_list.append(filename.replace(".gif",""))
            # print("parent is:" + parent)
            # print("filename is:" + filename)
            # print("the full name of the file is:" + os.path.join(parent, filename))  # 输出文件路径信息
    return img_list
img_list = gen_list()
def gen_captcha_text_and_image_new():
    img = choice(img_list)
    captcha_image = Image.open(root_dir + "/" + img + ".gif")
    captcha_image = np.array(captcha_image)
    return img, captcha_image


# if __name__ == '__main__':
#     # 测试
#     # text, image = gen_captcha_text_and_image()
#     #
#     # f = plt.figure()
#     # ax = f.add_subplot(111)
#     # ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
#     # plt.imshow(image)
#     # plt.show()
#     #
#
#     text, image = gen_captcha_text_and_image_new()
#
#     f = plt.figure()
#     ax = f.add_subplot(111)
#     ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
#     plt.imshow(image)
#     plt.show()
