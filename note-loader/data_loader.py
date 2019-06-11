import pretty_midi
import numpy as np
from utils import *

class MidiLoader:
    def __init__(self):
        '''
        initialization of dimensions
        we provide both duration and duration ratio for future comparison
        '''
        self.pitch_dim = 128
        self.ioi_dim = 100
        self.dur_dim = 500 # base 10 ms, from 0s to 5s
        # self.durratio_dim = 100 # base is 0.1, from 0 to 10
        self.dy_dim = 8
        self.onehot_dim = self.pitch_dim + self.ioi_dim + self.dur_dim + self.dy_dim
        # self.onehot_dim = self.pitch_dim + self.ioi_dim + self.durratio_dim + self.dy_dim
        # idx of each 1hot vector
        self.pitch_idx = 0
        self.ioi_idx = self.pitch_dim
        self.dur_idx = self.pitch_dim + self.ioi_dim
        self.dy_idx = self.pitch_dim + self.ioi_dim + self.dur_dim
        # self.duratio_idx = self.pitch_dim + self.ioi_dim
        # self.dy_idx = self.pitch_dim + self.ioi_dim + self.durratio_dim
        # base values
        self.dur_base = 0.01 # 10 ms
        self.durratio_base = 0.1


    def str_notes_to_1hot(self, song_folder, score_dir):
        '''
        For now, we assume that all txt files in song_folder are fake ground truth of the performance
        '''
        # load score
        score_file = find('*.txt', score_dir)
        score = np.loadtxt(score_dir + score_file[0])
        num_notes = score.shape[0]
        print(num_notes)
        # find all files in the folder
        files = find('*.txt', song_folder)
        num_files = len(files)
        notes_1hot = np.zeros((num_notes, self.onehot_dim, num_files))
        for i, file in enumerate(files):
            file = song_folder + file
            notes = np.loadtxt(file)
            notes_1hot[:, :, i] = self.str_notes_to_1hot_1song(notes, score)

    def str_notes_to_1hot_1song(self, notes, score):
        num_notes = notes.shape[0]
        notes_1hot = np.zeros((num_notes, self.onehot_dim))
        #### convert pitch to 1hot ####
        notes_1hot[:, self.pitch_idx:self.ioi_idx] = self.pitch_1hot_encode(notes[:, 0], num_notes)
        #### convert duration to 1hot ####
        dur = notes[:, 3] - notes[:, 2]
        score_dur = score[:, 3] - score[:, 2]
        # durratio = dur / score_dur
        notes_1hot[:, self.dur_idx:self.dy_idx] = self.dur_1hot_encode(dur, num_notes)
        #### convert dynamic to 1hot ####
        notes_1hot[:, self.dy_idx:] = self.dy_1hot_encode(notes[:, 1], num_notes)
        return notes_1hot

    def pitch_1hot_encode(self, pitches, num_notes):
        classes = np.arange(0, 128)
        pitch_1hot = np.zeros((num_notes, 128))
        pitch = pitches.astype(int) - 1 
        pitch_1hot[np.arange(pitch.shape[0]), pitch] = 1
        return pitch_1hot.astype(int)

    def dur_1hot_encode(self, dur, num_notes):
        classes = np.arange(0, self.dur_dim)
        dur_1hot = np.zeros((num_notes, self.dur_dim))
        # normalize duration by dividing base units
        # replace duration larger than 5s with 5s
        dur = (dur / self.dur_base).astype(int)
        c = np.where(dur >= self.dur_dim)
        dur[c] = self.dur_dim - 1
        dur_1hot[np.arange(num_notes), dur] = 1
        return dur_1hot.astype(int)
        
    def dy_1hot_encode(self, dy, num_notes):
        classes = np.arange(0, self.dy_dim)
        dy_1hot = np.zeros((num_notes, self.dy_dim))
        dy_norm = (dy / 16).astype(int) - 1
        dy_1hot[np.arange(num_notes), dy_norm] = 1
        return dy_1hot.astype(int)
        