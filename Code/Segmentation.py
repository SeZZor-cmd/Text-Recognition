import cv2
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
