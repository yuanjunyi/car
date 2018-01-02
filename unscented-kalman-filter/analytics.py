import sys
import numpy as np

if __name__ == '__main__':
	filename = sys.argv[1]
	with open(filename, 'r') as f:
		lines = f.read().splitlines()
		measurements = []
		for line in lines:
			words = line.split('\t')
			vx, vy, yaw = [float(word) for word in words[-4:-1]]
			measurements.append([vx, vy, yaw])

		delta_t = 0.05
		acceleration_vx = [(j[0]-i[0]) / 0.05 for i, j in zip(measurements[:-1], measurements[1:])]
		acceleration_vy = [(j[1]-i[1]) / 0.05 for i, j in zip(measurements[:-1], measurements[1:])]
		acceleration_yaw = [(j[2]-i[2]) / 0.05 for i, j in zip(measurements[:-1], measurements[1:])]
		
		print(np.std(acceleration_vx), np.max(acceleration_vx))
		print(np.std(acceleration_vy), np.max(acceleration_vy))
		print(np.std(acceleration_yaw), np.max(acceleration_yaw))
