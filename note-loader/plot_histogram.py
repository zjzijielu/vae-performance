import numpy as np
from os import listdir
from os.path import isfile, join
from utils import *
from collections import Counter
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
song_folder = parent_dir + "/data/perf_start_sorted/"

file_list = find('*.csv', song_folder)

d = []

for file_name in file_list:
    mat = np.loadtxt(song_folder + file_name, delimiter=',')
    try:
        d.extend(mat[:, 4].astype(int))
        # dynamic_counter += dynamic
    except:
        print(len(d))
        continue

mean = np.mean(d)
std = np.std(d)        

n, bins, patches = plt.hist(x=d, bins=128, color='#607c8e', rwidth=0.9)
plt.grid(axis='y')
plt.xlabel("Dynamic \n mean= %.2f std= %.2f" % (mean, std))
plt.ylabel('Frequency')
plt.title('Dynamic frequency')
maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 + 10000 if maxfreq % 10 else maxfreq + 10000)

plt.show()