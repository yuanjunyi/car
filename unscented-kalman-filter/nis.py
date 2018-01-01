import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
	filename = sys.argv[1]
	with open(filename, 'r') as f:
		lines = f.readlines()
		nis = [float(x) for x in lines[2:]]
		plt.plot(nis)
		if filename == 'r.txt':
			plt.hlines(7.815, 0, len(nis))
		else:
			plt.hlines(5.991, 0, len(nis))
		plt.show()

