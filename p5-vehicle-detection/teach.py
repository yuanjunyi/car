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
	for x in range(xmin, xmax, window):
		for y in range(ymin, ymax, window):
			subimg = img[y:min(y+window,ymax), x:min(x+window,xmax)]
			h, w = subimg.shape[:2]
			if w == 64 and h == 64:
				mpimg.imsave('teach_images/%d.png' % count, subimg)
				count += 1
			if w > 64 and h > 64:
				subimg = cv2.resize(subimg, (64, 64))
				mpimg.imsave('teach_images/%d.png' % count, subimg)
				count += 1

if __name__ == '__main__':
	filenames = [
		'test_images/test1.jpg',
		'test_images/test2.jpg',
		'test_images/test3.jpg',
		'test_images/test4.jpg',
		'test_images/test5.jpg',
		'test_images/test6.jpg',
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
	]
	for filename in filenames:
		for delta in range(0, 64, 8):
			for window in range(64, 280, 8):
				extract(filename, window, xmin=0+delta, xmax=800, ymin=400+delta, ymax=700)
				extract(filename, window, xmin=700+delta, xmax=1000, ymin=500+delta, ymax=700)