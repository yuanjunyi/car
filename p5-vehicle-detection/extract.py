import sys
import cv2
import csv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

count = 0

def extract(filename, window, xmin, xmax, ymin, ymax):
	global count
	img = mpimg.imread(filename)
	for x in range(xmin, xmax, 16):
		for y in range(ymin, ymax, 16):
			subimg = img[y:min(y+window,ymax), x:min(x+window,xmax)]
			h, w = subimg.shape[:2]
			if w == 64 and h == 64:
				mpimg.imsave('non-vehicles/Video_extracted/%d.png' % count, subimg)
				count += 1
			if w > 64 and h > 64:
				subimg = cv2.resize(subimg, (64, 64))
				mpimg.imsave('non-vehicles/Video_extracted/%d.png' % count, subimg)
				count += 1

if __name__ == '__main__':
	filenames = [
		'test_images/test07.jpg',
		'test_images/test08.jpg',
		'test_images/test09.jpg',
		'test_images/test10.jpg',
		'test_images/test11.jpg',
		'test_images/test12.jpg',
		'test_images/test13.jpg',
		'test_images/test14.jpg',
		'test_images/test15.jpg',
		'test_images/test16.jpg',
		'test_images/test17.jpg',
		'test_images/test231.jpg',
		'test_images/test232.jpg',
		'test_images/test233.jpg',
		'test_images/test234.jpg',
		'test_images/test235.jpg'
	]
	for filename in filenames:
		print('extracting', filename)
		for window in (64, 96, 128):
			extract(filename, window, xmin=0, xmax=1200, ymin=450, ymax=700)