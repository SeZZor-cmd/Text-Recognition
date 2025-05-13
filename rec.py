import os
import cv2
import pandas as pd
from Segmentation import dp

dir_path = r'D:\\Study\\FINAL PROJECT\\Test Img'
res = []
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)

path=dp
image=cv2.imread(path)
import ntpath
name=ntpath.basename(path)
ind = res.index(name)
df = pd.read_csv('word.csv', encoding='utf-16')
for index, row in df.iterrows():
    if row['class'] == ind :
        print(row['char'])