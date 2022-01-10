# -*- coding: utf-8 -*-
# @Time    : 2019-05-21 19:55
# @Author  : LeeHW
# @File    : Prepare_data.py
# @Software: PyCharm
from glob import glob
from flags import *
import os
import cv2
import numpy as np
import datetime
from multiprocessing.dummy import Pool as ThreadPool

starttime = datetime.datetime.now()

save_HR_path = os.path.join(save_dir, 'HR_x3')
save_LR_path = os.path.join(save_dir, 'LR_x3')
os.mkdir(save_HR_path)
os.mkdir(save_LR_path)
file_list = sorted(glob(os.path.join(train_HR_dir, '*.png')))
HR_size = [1, 0.8, 0.7, 0.6, 0.5]


def save_HR_LR(img, size, path, idx):
	h, w, _ = img.shape
	HR_img = cv2.resize(img, (int(w * size), int(h * size)), interpolation=cv2.INTER_CUBIC)
	HR_img = modcrop(HR_img, 3)
	
	h1, w1, _ = HR_img.shape
	rot180_img = cv2.rotate(HR_img, cv2.ROTATE_180)
	x3_img = cv2.resize(HR_img, (w1 // 3, h1 // 3), interpolation=cv2.INTER_CUBIC)
	x3_rot180_img = cv2.resize(rot180_img, (w1 // 3, h1 // 3), interpolation=cv2.INTER_CUBIC)
	hr = path.split('/')[-1].split('.')[0]
	lr = path.split('/')[-1].split('.')[0]

	img_path = hr[15 :]+ '_rot0_' + 'ds' + str(idx) + '.png'
	rot180img_path = hr[15 :] + '_rot180_' + 'ds' + str(idx) + '.png'
	x3_img_path = lr[15 :] + '_rot0_' + 'ds' + str(idx) + '.png'
	x3_rot180img_path = lr[15 :] + '_rot180_' + 'ds' + str(idx) + '.png'

	cv2.imwrite(os.path.join(save_HR_path, img_path), HR_img)
	cv2.imwrite(os.path.join(save_HR_path, rot180img_path), rot180_img)
	cv2.imwrite(os.path.join(save_LR_path, x3_img_path), x3_img)
	cv2.imwrite(os.path.join(save_LR_path, x3_rot180img_path), x3_rot180_img)
	cv2.waitKey(0)


def modcrop(image, scale=4):
	if len(image.shape) == 3:
		h, w, _ = image.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		image = image[0:h, 0:w, :]
	else:
		h, w = image.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		image = image[0:h, 0:w]
	return image


def main(path):
	img = cv2.imread(path)
	idx = 0
	for size in HR_size:
		save_HR_LR(img, size, path, idx)
		idx += 1

items = file_list
pool = ThreadPool()
pool.map(main, items)
pool.close()
pool.join()

