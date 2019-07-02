from os import listdir
from os.path import isfile, join
from utils import *
import csv

def normalize_ioi(mat):
    '''
    TODO
    '''
    c = np.where(mat[:, -2] > 8)[0]
    if c.size == 0:
        return mat
    else:
        print(c)
        print(mat[c, -2])
        start_beat = mat[c, 1]
        start_time = mat[c, 5]
        raise ValueError
    
    return mat

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
perf_sorted_dir = parent_dir + '/data/perf_start_sorted/'
song_folder = parent_dir + "/data/perf_tempo_csv/"

file_list = find('*.csv', song_folder)

for i in range(len(file_list)):
    file_name = file_list[i]
    print(file_name)
    f = np.loadtxt(song_folder + file_name, delimiter=',')
    track_start_indices = []
    num_tracks = f[-1, 0] + 1
    for i in range(int(num_tracks)):
        indices = np.where(f[:, 0] == i)[0]
        # print(indices)
        track_start_indices.append(int(indices[0]))
    
    for i in range(int(num_tracks)):
        start = track_start_indices[i]
        if (i < num_tracks - 1):
            end = track_start_indices[i + 1]
            single_track = f[start:end, :]
        else:
            single_track = f[start:, :]
        single_track = single_track[single_track[:, 5].argsort()]
        # print(single_track[:, 5])

        # add two extra columns for ioi
        row, col = single_track.shape
        new_mat = np.zeros((row, col + 2))
        new_mat[:, :-2] = single_track
        new_mat[0, -2] = new_mat[0, 1]
        new_mat[0, -1] = new_mat[0, 5]
        new_mat[1:, -2] = new_mat[1:, 1] - new_mat[0:-1, 1]
        new_mat[1:, -1] = new_mat[1:, 5] - new_mat[0:-1, 5]

        c = np.where(new_mat < 0)
        assert(c[0].size == 0)

        # normalize ioi
        # new_mat = normalize_ioi(new_mat)
        np.savetxt(perf_sorted_dir + file_name[:-4] + '_' + str(i) + ".csv", new_mat, delimiter=',', fmt="%4f")

    # print(num_tracks)
    