import toml
import os
import cv2
import numpy as np

dict_toml = toml.load(open('../config.toml'))
cascade_path = dict_toml["path"]["cascades"] + "/haarcascade_frontalface_default.xml"


image_path = dict_toml["path"]["dataset"]["image"]+"/img_align_celeba/" 
output_path = os.getcwd()+"/image/"
print(output_path)

image_files = os.listdir(dict_toml["path"]["dataset"]["image"]+"/img_align_celeba/")

datalist = []

for i in image_files:
    image = cv2.imread(image_path+i)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
    
    if len(facerect) == 1:
        
        mannaka = np.flip(np.ceil(facerect[0][0:2]+facerect[0][2:4]/2))
        h = image.shape[:2] - np.flip(np.ceil(facerect[0][0:2]+facerect[0][2:4]/2))
        m = np.minimum(mannaka, h)
        
        top = mannaka - m.min()
        bot = mannaka + m.min()
        
        img2 = image[int(top[0]):int(bot[0]),int(top[1]):int(bot[1])]
        
        img3 = cv2.resize(img2, dsize=(32, 32))
        
        cv2.imwrite(output_path+i, img3)
        
    else:
        print("{} is not　detected face.".format(i))
        datalist.append(i+"\n")


f = open('not_detected.txt', 'w')

f.writelines(datalist)

f.close()