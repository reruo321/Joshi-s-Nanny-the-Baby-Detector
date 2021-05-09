import cv2
import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import time
import tkinter
import tkinter.font
from tkinter import filedialog

motion_frame = []

baby_num = 0
person_num = 0
frame_num = 0
baby_moving_num = 0

def box_motion_detector(image, left, right, top, bottom):
    global motion_frame, baby_num, baby_moving_num

    if image is None or left is None or right is None or top is None or bottom is None:
        return 0

    img_area = image.crop((left, top, right, bottom))
    img_area = np.array(img_area)
    gray = cv2.cvtColor(img_area, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21),0)

    baby_num = baby_num + 1

    if baby_num > 1:
        motion_frame.append(gray)
        if motion_frame[0].shape[0] != motion_frame[1].shape[0] or motion_frame[0].shape[1] != motion_frame[1].shape[1]:
            if abs(motion_frame[0].shape[0] - motion_frame[1].shape[0]) >= 5 or abs(motion_frame[0].shape[1] - motion_frame[1].shape[1]) >= 5:
                baby_moving_num = baby_moving_num + 1
                motion_frame.clear()
                motion_frame.append(gray)
                return 1
            else:
                motion_frame.clear()
                motion_frame.append(gray)
                return 0
        else:
                delta = cv2.absdiff(motion_frame[0], motion_frame[1], gray)
                thresh = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                (cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                motion_frame.clear()
                motion_frame.append(gray)

                for contour in cnts:
                    if cv2.contourArea(contour) < 100:
                        return 0
                    else:
                        baby_moving_num = baby_moving_num + 1
                        return 1
    else:
        motion_frame.append(gray)
        return 0

def display_image(image):
  global frame_num
  # release some memory from image buffer
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.clf()
  plt.close()

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               isBaby,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)

  if isBaby == 1:
    if box_motion_detector(image, left, right, top, bottom):
      color = "rgb(255, 0, 0)" # moving_baby

  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height

  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=100, min_score=0.4):
  """Overlay labeled boxes on an image with formatted scores and label names."""

  font = ImageFont.load_default()
  global person_num

  for i in range(1,3):

      class_12 = None
      isBaby = 0
      color = "rgb(0, 0, 0)" # default

      for j in range(max_boxes):
        if scores[0][j] >= min_score:
            if class_names[0][j] == 1. and i == 1.:
                color = "rgb(255, 255, 255)" # person
                isBaby = 0
                person_num = person_num + 1
                class_12 = "Person"
            elif class_names[0][j] == 2. and i == 2.:
                color = "rgb(0, 255, 0)" # not_moving_baby
                isBaby = 1
                class_12 = "Baby"
            else:
                continue
        if class_12 is None:
            break
        ymin, xmin, ymax, xmax = tuple(boxes[0][j])
        display_str = "{}: {}%".format(class_12, int(100 * scores[0][j]))

        image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
        draw_bounding_box_on_image(
               image_pil,
               ymin,
               xmin,
               ymax,
               xmax,
               color,
               font,
               isBaby,
               display_str_list=[display_str])
        np.copyto(image, np.array(image_pil))
        break
  return image

def load_img(frame_img):
  tf_frame_img = tf.convert_to_tensor(frame_img)
  del frame_img
  return tf_frame_img

def run_detector(detector, img):
  global baby_num, baby_moving_num, person_num, frame_num
  loaded_img = load_img(img)
  converted_img  = tf.image.convert_image_dtype(loaded_img, tf.uint8)[tf.newaxis, ...]

  start_time = time.time()

  result = detector(converted_img)
  end_time = time.time()

  result = {key:value.numpy() for key,value in result.items()}

  print("Inference time: ", end_time-start_time)

  image_with_boxes = draw_boxes(
      img, result["detection_boxes"],
      result["detection_classes"], result["detection_scores"])

  frame_num = frame_num + 1
  display_image(image_with_boxes)
  del result
  del converted_img
  print('Frame Info:\nBaby = {} | Moving_Baby = {} | Person = {} | Total = {}\n'.format(str(baby_num), str(baby_moving_num), str(person_num), str(frame_num)))

saved_model_path = r'.\joshi_model_v3\saved_model'
detector = hub.load(saved_model_path)

video_flag = 0

def load_video():
    global video_flag
    video_flag = filedialog.askopenfilename(initialdir = '/', title='Select Video File', filetypes=(('MP4', '*.mp4'), ('All files', '*.*')))
    scr.destroy()

def load_webcam():
    global video_flag
    video_flag = 0
    scr.destroy()

scr = tkinter.Tk()
scr.title("Joshi's Nanny")
scr.geometry("640x480")
font = tkinter.font.Font(size=50)
vid = tkinter.Button(scr, text='Load Video', overrelief="solid", background='orange', font=font, command=load_video, repeatdelay=1000)
webc = tkinter.Button(scr, text='Load Webcam', overrelief="solid", background='green', font=font, fg='white', command=load_webcam, repeatdelay=1000)
vid.pack(fill='both', expand=True)
webc.pack(fill='both', expand=True)

scr.mainloop()

video = cv2.VideoCapture(video_flag)

#    video = cv2.VideoCapture(r'.\joshi_sleeping.mp4')
#    video = cv2.VideoCapture(0)

while True:

    check, frame = video.read()

    if not check:
        break

    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_img = frame.copy()

    run_detector(detector, frame_img)

    cv2.imshow("Joshi's Nanny", frame_img)

# Closes all windows
cv2.destroyAllWindows()

# Releases video file/webcam
video.release()
