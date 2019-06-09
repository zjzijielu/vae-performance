import pretty_midi
import numpy as np
from utils import *

class MidiLoader:
    def __init__(self):
        self.pitch_dim = 128
        self.ioi_dim = 100
        self.dur_dim = 200
        self.dy_dim = 5
        self.onehot_dim = self.pitch_dim + self.ioi_dim + self.dur_dim + self.dy_dim
    
    def str_notes_to_1hot(self, song_folder):
        # find all files in the folder
        files = find('*.txt', song_folder)
        for i, file in enumerate(files):
            file = song_folder + file
            song_matrix = np.loadtxt(file)
            print(song_matrix)
