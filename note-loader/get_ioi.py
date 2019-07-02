from os import listdir
from os.path import isfile, join
from utils import *
import csv

perf_ioi_dir = '/Users/luzijie/Desktop/vae-performance/data/perf_with_ioi_csv/'

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
song_folder = parent_dir + "/data/perf_start_sorted/"

ioi_max_t = 0.5
ioi_max_beat = 3

file_list = find('*.csv', song_folder)

for i in range(len(file_list)):
    file_name = file_list[i]
    f = np.loadtxt(song_folder + file_name, delimiter=",")
    max_t = np.amax(f[1:, -1])
    max_beat = np.amax(f[1:, -2])

    larger_than_4 = np.where(f[0, -1] > 4)[0]
    print("larger_than_4 ratio:", larger_than_4.size / f.shape[0])

    min_t = np.amin(f[1:, -1])
    if (min_t < 0.1 and min_t > 0):
        print(file_name)
        print("min_t:", min_t)

    if (ioi_max_t < max_t and max_t < 10):
        ioi_max_t = max_t
        print(file_name)
        print("ioi_max_t:", ioi_max_beat)
    if (ioi_max_beat < max_beat and max_beat < 8):
        ioi_max_beat = max_beat
        print(file_name)
        print("ioi_max_beat:", ioi_max_beat)
    

print("ioi_max_t:", ioi_max_t)
print("ioi_max_beat:", ioi_max_beat)
                