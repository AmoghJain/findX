from flask import Flask
import json
from flask import request
import cv2
import base64
import tensorflow as tf
import numpy as np
import model

app = Flask(__name__)

@app.route("/", methods=["POST"])
def home():
    input_json = json.loads(request.data)
    str_to_img(input_json["image"])
    return "Success!"


def str_to_img(img_str):
    data = base64.b64decode(img_str)
    image_file = open("test.jpg", "wb")
    image_file.write(data)
    image_file.close()
    img = cv2.imread("test.jpg", 0)
    process(img)


def process(image):
    ret, thresh = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
    inverted = (255 - thresh)
    resized = cv2.resize(inverted, (28, 28))
    # cropped = get_cropped(inverted)
    cv2.imwrite("test1.jpg", resized)
    model.predict(resized)


# def get_cropped(image):
#     h_cropped = get_horizontal_cropped(image)
#     v_cropped = get_vertical_cropped(h_cropped)
#
#
# def get_horizontal_cropped(image):
#     top = 0
#     bottom = 0
#     for i in range(len(image)):
#         if max(image[i]) == 255:
#             top = i
#             break
#     for i in range(len(image[::-1])):
#         if max(image[i]) == 255:
#             bottom = len(image) - i
#             break
#     cropped_image = image[top:bottom+10, :]
#     cv2.imwrite("test.jpg", cropped_image)
#     return cropped_image
#
#
# def get_vertical_cropped(image):
#     left = 0
#     right = 0
#     # cv2.imwrite("test.jpg", cropped_image)
