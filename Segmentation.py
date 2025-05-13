"""import cv2
import numpy as np
import matplotlib.pyplot as plt
#from config import address
import os
dp=r"D:\Study\FINAL PROJECT\Test Img\T2.jpeg"
img = cv2.imread(dp)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, c = img.shape

if w > 1000:
    new_w = 1000
    ar = w / h
    new_h = int(new_w / ar)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
plt.imshow(img)
plt.show()

def thresholding(image):
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
    plt.imshow(thresh, cmap='gray')
    plt.show()
    return thresh

thresh_img = thresholding(img)

def invert(image):
    inverted_image = cv2.bitwise_not(image)
    plt.imshow(inverted_image)
    plt.show()
    return inverted_image

inverted=invert(img)

#dilation
kernel = np.ones((3,90), np.uint8)
dilated = cv2.dilate(thresh_img, kernel, iterations = 1)
plt.imshow(dilated, cmap='gray')
plt.show()

(contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h)

img2 = img.copy()

for ctr in sorted_contours_lines:
    x, y, w, h = cv2.boundingRect(ctr)
    cv2.rectangle(img2, (x, y), (x + w, y + h), (40, 100, 250), 2)

plt.imshow(img2)
plt.show()

#dilation
kernel = np.ones((3,1), np.uint8)
dilated2 = cv2.dilate(thresh_img, kernel, iterations = 1)
plt.imshow(dilated2, cmap='gray')

img3 = img.copy()
words_list = []

for line in sorted_contours_lines:

    # roi of each line
    x, y, w, h = cv2.boundingRect(line)
    roi_line = dilated2[y:y + w, x:x + w]

    # draw contours on each word
    (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contour_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])

    for word in sorted_contour_words:

        if cv2.contourArea(word) < 400:
            continue

        x2, y2, w2, h2 = cv2.boundingRect(word)

        words_list.append([x + x2, y + y2, x + x2 + w2, y + y2 + h2])
        cv2.rectangle(img3, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255, 255, 100), 2)

plt.imshow(img3)
plt.show()
x = len(words_list)
print((x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255, 255, 100), 2)

word = words_list[0]
roi_9 = img[word[1]:word[3],word[0]:word[2]]
plt.imshow(roi_9)
plt.show()
print(word)

from PIL import Image
i=0
path='D:\Study\FINAL PROJECT\out'
out=[]
for i in range(x):
    word = words_list[i]
    roi = img[word[1]:word[3],word[0]:word[2]]
    im = Image.fromarray(roi)
    im.save(f"{path}/ch"+str(i+1)+".jpg")
    filename='ch'+str(i+1)+'.jpg'
    out.append(filename)

import keras
import numpy as np



import rec
def predict(out):
    image_paths = out
    print(image_paths)
    output = []
    for i in image_paths:
        m = []
        image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("image",image)
        image = cv2.resize(image, (32, 32))
        image = np.reshape(image, (32, 32, 1)) / 255
        m.append(image)
        m = np.array(m)

        y_classes = np.argmax(model.predict(m))
        output.append(catergories[y_classes])
    return output


# Considering y variable holds numpy array

model = keras.models.load_model('model.weights.best.hdf5')
CATEGORIES = ['ଅ', 'ଆ', 'ଇ', 'ଈ', 'ଉ', 'ଊ', 'ଋ', 'ଏ', 'ଐ', 'ଓ', 'ଔ', 'କ',
              'ଖ', 'ଗ', 'ଘ', 'ଙ', 'ଚ', 'ଛ', 'ଜ', 'ଝ', 'ଞ', 'ଟ', 'ଠ', 'ଡ', 'ଢ',
              'ଣ', 'ତ', 'ଥ', 'ଦ', 'ଧ', 'ନ', 'ପ', 'ଫ', 'ବ', 'ଭ', 'ମ', 'ଯ', 'ର',
              'ଲ', 'ଳ', 'ଶ', 'ଷ', 'ସ', 'ହ', 'ୟ']

catergories = np.array(CATEGORIES)
image_paths = path
for i, image_path in enumerate(out):
    #     y_tensor = tf.convert_to_tensor(image_path, dtype=tf.int64)
    answer = (image_path)
    (''.join(answer))  # will be the output string
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from keras.models import load_model
import tensorflow as tf

dp = r"D:\Study\FINAL PROJECT\Test Img\T2.jpeg"


def load_and_resize_image(dp, max_width=1000):
    img = cv2.imread(dp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    if w > max_width:
        new_w = max_width
        ar = w / h
        new_h = int(new_w / ar)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def thresholding(image, threshold_value=80):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return thresh


def invert(image):
    return cv2.bitwise_not(image)


def find_contours(image, kernel_size=(3, 90), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    return sorted_contours


def draw_rectangles(image, contours):
    img_copy = image.copy()
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (40, 100, 250), 2)
    return img_copy


def process_words(image, contours, kernel_size=(3, 1), min_area=400):
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    words_list = []
    img_copy = image.copy()

    for line in contours:
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated[y:y + h, x:x + w]

        cnt, _ = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contour_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])

        for word in sorted_contour_words:
            if cv2.contourArea(word) < min_area:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(word)
            words_list.append([x + x2, y + y2, x + x2 + w2, y + y2 + h2])
            cv2.rectangle(img_copy, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255, 255, 100), 2)

    return img_copy, words_list


def save_word_images(image, words_list, path='D:\Study\FINAL PROJECT\out'):
    if not os.path.exists(path):
        os.makedirs(path)
    out = []
    for i, word in enumerate(words_list):
        roi = image[word[1]:word[3], word[0]:word[2]]
        im = Image.fromarray(roi)
        filename = f"{path}/ch{i + 1}.jpg"
        im.save(filename)
        out.append(filename)
    return out


def predict_characters(image_paths, model, categories):
    output = []
    for i in image_paths:
        image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (32, 32))
        image = np.reshape(image, (32, 32, 1)) / 255.0
        image = np.expand_dims(image, axis=0)

        y_classes = np.argmax(model.predict(image))
        output.append(categories[y_classes])
    return output


# Main Code Execution
img = load_and_resize_image(dp)
plt.imshow(img)
plt.show()

thresh_img = thresholding(img)
plt.imshow(thresh_img, cmap='gray')
plt.show()

inverted_img = invert(img)
plt.imshow(inverted_img)
plt.show()

contours = find_contours(thresh_img)
img_with_rectangles = draw_rectangles(img, contours)
plt.imshow(img_with_rectangles)
plt.show()

img_with_words, words_list = process_words(thresh_img, contours)
plt.imshow(img_with_words)
plt.show()

num_words = len(words_list)
print(f"Number of words detected: {num_words}")

word_images = save_word_images(img, words_list)

model_path = 'model.weights.best.hdf5'
model = load_model(model_path)
categories = ['ଅ', 'ଆ', 'ଇ', 'ଈ', 'ଉ', 'ଊ', 'ଋ', 'ଏ', 'ଐ', 'ଓ', 'ଔ', 'କ',
              'ଖ', 'ଗ', 'ଘ', 'ଙ', 'ଚ', 'ଛ', 'ଜ', 'ଝ', 'ଞ', 'ଟ', 'ଠ', 'ଡ', 'ଢ',
              'ଣ', 'ତ', 'ଥ', 'ଦ', 'ଧ', 'ନ', 'ପ', 'ଫ', 'ବ', 'ଭ', 'ମ', 'ଯ', 'ର',
              'ଲ', 'ଳ', 'ଶ', 'ଷ', 'ସ', 'ହ', 'ୟ']

predictions = predict_characters(word_images, model, categories)
print(''.join(predictions))
