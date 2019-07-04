'''
This file stores all necessary parameters
'''

class parameters():
    def __init__(self):
        # score part
        self.pitch_dim = 128
        self.ioi_beat_dim = 128 # base 1/16, longest 8 beats
        self.dur_dim = 64 # base 1/16 beat, longest 4 beats

        # performance part
        self.durratio_dim = 250 # base is 0.02, from 0 to 5
        self.dy_dim = 8
        self.ioi_time_dim = 801 # base is 0.01s, maximum 8s

        # total dimension
        self.score_1hot_dim = self.pitch_dim + self.ioi_beat_dim + self.dur_dim
        self.perf_1hot_dim = self.durratio_dim + self.dy_dim + self.ioi_time_dim
        self.onehot_dim = self.score_1hot_dim + self.perf_1hot_dim

        # idx of each 1hot vector
        self.pitch_idx = 0
        self.ioi_beat_idx = self.pitch_dim
        self.dur_idx = self.ioi_beat_idx + self.ioi_beat_dim

        self.durratio_idx = self.score_1hot_dim
        self.dy_idx = self.durratio_idx + self.durratio_dim
        self.ioi_time_idx = self.dy_idx + self.dy_dim

        # base values
        self.dur_base = 0.0625 # 1/16 beat
        self.ioi_beat_base = 0.0625
        self.durratio_base = 0.02
        self.ioi_time_base = 0.01
        self.max_dur_beat = 8
        self.min_notes_num = 3