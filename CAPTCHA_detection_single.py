import time
import tensorflow as tf
import time
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


PATH_TO_SAVED_MODEL="inference_graph\\saved_model\\"


print('Loading model...', end='')


detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)

print('Model Loaded!')

category_index=label_map_util.create_category_index_from_labelmap("labelmap.pbtxt",use_display_name=True)


def load_image_into_numpy_array(path):

    return np.array(Image.open(path))



image_path = '1BseN.jpg'

image_np = load_image_into_numpy_array(image_path)

input_tensor = tf.convert_to_tensor(image_np)

input_tensor = input_tensor[tf.newaxis, ...]

detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}

boxes=detections['detection_boxes']
classes=detections['detection_classes']
scores=detections['detection_scores']

captcha_array = []
for i,b in enumerate(boxes):
    for Symbol in range(63):
        if classes[i] == Symbol:
            if scores[i] >= 0.30:
                mid_x = (boxes[i][1]+boxes[i][3])/2
                captcha_array.append([category_index[Symbol].get('name'), mid_x, scores[i]])

    #rechecking output

if len(captcha_array) <5:
    captcha_array = []
    for i,b in enumerate(boxes):
        for Symbol in range(63):
            if classes[i] == Symbol:
                if scores[i] >= 0.20:
                    mid_x = (boxes[i][1]+boxes[i][3])/2
                    captcha_array.append([category_index[Symbol].get('name'), mid_x, scores[i]])

if len(captcha_array) <5:
    captcha_array = []
    for i,b in enumerate(boxes):
        for Symbol in range(63):
            if classes[i] == Symbol:
                if scores[i] >= 0.10:
                    mid_x = (boxes[i][1]+boxes[i][3])/2
                    captcha_array.append([category_index[Symbol].get('name'), mid_x, scores[i]])


for number in range(20):
    for captcha_number in range(len(captcha_array)-1):
        if captcha_array[captcha_number][1] > captcha_array[captcha_number+1][1]:
            temporary_captcha = captcha_array[captcha_number]
            captcha_array[captcha_number] = captcha_array[captcha_number+1]
            captcha_array[captcha_number+1] = temporary_captcha

average_distance_error=3
average = 0
captcha_len = len(captcha_array)-1
while captcha_len > 0:
    average += captcha_array[captcha_len][1]- captcha_array[captcha_len-1][1]
    captcha_len -= 1


average = average/(len(captcha_array)+average_distance_error)
captcha_array_filtered = list(captcha_array)
captcha_len = len(captcha_array)-1
while captcha_len > 0:
    if captcha_array[captcha_len][1]- captcha_array[captcha_len-1][1] < average:
        if captcha_array[captcha_len][2] > captcha_array[captcha_len-1][2]:
            del captcha_array_filtered[captcha_len-1]
        else:
            del captcha_array_filtered[captcha_len]
    captcha_len -= 1

    # Get final string from filtered CAPTCHA array
captcha_string = ""
for captcha_letter in range(len(captcha_array_filtered)):
    captcha_string += captcha_array_filtered[captcha_letter][0]


print(captcha_string)

