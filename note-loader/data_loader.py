import pretty_midi
import numpy as np
from os import listdir
from os.path import isfile, join
from utils import *
import random

perf_score_dir = '/Users/luzijie/Desktop/vae-performance/data/perf_score/'
perf_tempo_dir = '/Users/luzijie/Desktop/vae-performance/data/perf_tempo/'

class MidiLoader:
    def __init__(self, least_change=3):
        '''
        initialization of dimensions
        we provide both duration and duration ratio for future comparison
        '''
        # score part
        self.pitch_dim = 128
        self.ioi_beat_dim = 128 # base 1/16, longest 8 beats
        self.dur_dim = 64 # base 1/16 beat, longest 4 beats

        # performance part
        self.durratio_dim = 250 # base is 0.02, from 0 to 5
        self.dy_dim = 64
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
        self.least_change = least_change
        self.max_dur_beat = 8
        self.min_notes_num = 3
        self.dy_base = 128 / self.dy_dim

        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.output_dir = parent_dir + "/data/2bars_data/"
        self.perf_midi_dir = parent_dir + "/data/perf_midi_syn/"
        self.score_midi_dir = parent_dir + "/data/score_midi_syn/"

        assert(self.ioi_beat_idx - self.pitch_idx == self.pitch_dim)
        assert(self.dur_idx - self.ioi_beat_idx == self.ioi_beat_dim)
        assert(self.dy_idx - self.durratio_idx == self.durratio_dim)

    def str_notes_to_1hot(self, song_folder):
        '''
        input: 
            directory of raw performance data
        output: 
            2-bar-long performances data stored in "vae-perforamance/data/2bars-data/"
        
        in each row, the values in order are as follows:
        0. track
        1. start beat
        2. end beat
        3. pitch
        4. dynamic
        5. start time
        6. start tempo
        7. end time
        8. end tempo
        9. duration
        10. beat duration
        '''
        file_list = find('*.csv', song_folder)
        data_num = 0
        index = 0
        tempo_max = 200
        tempo_min = 70

        for i in range(len(file_list)):
            song_name = file_list[i]
            print(song_name)
            notes = np.loadtxt(song_folder + file_list[i], delimiter=',')
            # the notes are already seperated with respect to track
            notes_2bars = self.seperate_by_2bars(notes)
            ## test if synthesized midi makes sense
            # self.str_notes_to_midi(notes, file_list[i][:-10])
            # select 2-bar period with significant tempo change
            for n in notes_2bars:
                # convert the notes into 1hot vectors
                # print(n.shape)
                c = np.where(n[:, -2] > 8)[0]
                if c.size > 1 or (c.size == 1 and 0 not in c):
                    continue
                c = np.where(n[:, -1] > 8)[0]
                if c.size > 1 or (c.size == 1 and 0 not in c):
                    continue
                self.str_notes_to_1hot_1song(n, song_name, index)
                data_num += 1
                index += 1
            index = 0
        print("total 2-bar pieces:", data_num)
    
    def seperate_by_2bars(self, notes):
        '''
        input: 
            notes of a track
        output:
            list of 2-bar-long notes. In each row we omit the track number
        '''
        start_idx = 0
        start_beat = 0
        notes_2bars = []
        for i in range(1, len(notes)):
            if notes.ndim == 2:
                if (notes[i, 1] % self.max_dur_beat == 0 and start_beat != notes[i, 1]) or notes[i, 1] - notes[start_idx, 1] > self.max_dur_beat or notes[i, 1] // self.max_dur_beat != notes[start_idx, 1] // self.max_dur_beat:
                    # print("number of notes in 2 bars:", i - start_idx, "start:", start_idx + 1, "end:", i)
                    if i - start_idx >= self.min_notes_num:
                        notes_2bars.append(notes[start_idx:i, 1:])
                    start_idx = i
                    start_beat = notes[i, 1]
        notes_2bars = self.filter_by_tempo(notes_2bars, self.least_change)
        return notes_2bars

    def filter_by_tempo(self, notes_2bars, least_change):
        '''
        input:
            a list of 2-bar-long notes
        output:
            2-bar-long notes with both significant and insignificatn tempo change
            the ratio is 3:1
        
        if not specified, the standard that we filter with is that there has to be at least 3 tempo changes
        '''
        filtered = []
        num = 0
        for n in notes_2bars:
            start_tempo = n[:, 5]
            if len(np.unique(start_tempo)) >= least_change:
                filtered.append(n)
                num += 1
            else:
                if num % 3 == 0:
                    filtered.append(n)
                    num += 1

        return filtered
            
        

    def str_notes_to_1hot_1song(self, notes, song_name, index):
        '''
        in each row, the values in order are as follows:
        0. start beat
        1. end beat
        2. pitch
        3. dynamic
        4. start time
        5. start tempo
        6. end time
        7. end tempo
        8. duration
        9. beat duration
        10. score ioi
        11. perf ioi
        '''
        assert(notes.shape[1] == 12)

        # print(notes)
        num_notes = notes.shape[0]
        notes_1hot = np.zeros((num_notes, self.onehot_dim))
        #### convert pitch to 1hot ####
        notes_1hot[:, self.pitch_idx:self.ioi_beat_idx] = self.pitch_1hot_encode(notes[:, 2], num_notes)
        #### convert ioi beat to 1hot ####
        # print(notes[:, -2])
        notes_1hot[:, self.ioi_beat_idx:self.dur_idx] = self.ioi_beat_encode(notes[:, -2], num_notes)
        #### convert duration to 1hot ####
        notes_1hot[:, self.dur_idx:self.durratio_idx] = self.dur_1hot_encode(notes[:, 9], num_notes)
        #### convert durratio to 1hot ####
        # TODO: now our durratio is defined as perf_dur_time / score_dur_beats
        durratio = notes[:, 8] / notes[:, 9]
        # print("unique durratio:", len(np.unique(durratio)))
        notes_1hot[:, self.durratio_idx:self.dy_idx] = self.durratio_1hot_encode(durratio, num_notes)
        #### convert dynamic to 1hot ####
        notes_1hot[:, self.dy_idx:self.ioi_time_idx] = self.dy_1hot_encode(notes[:, 3], num_notes)
        #### convert tempo to 1hot ####
        notes_1hot[:, self.ioi_time_idx:] = self.ioi_time_encode(notes, num_notes)
        
        np.save(self.output_dir + song_name[:-4] + "_" + str(index), notes_1hot)
        return notes_1hot

    def str_notes_to_midi(self, notes, song):
        '''
        input:
            notes as numpy array
        output:
            synthesized midi for each track
        '''
        track_indicies = self.get_track_index(notes)
        for i in range(len(track_indicies) - 1):
            start = track_indicies[i]
            end = track_indicies[i + 1]
            subnotes = notes[start:end, :]
            # print(subnotes.shape)
            self.matrix2midi(subnotes, song, i)

    def matrix2midi(self, matrix, song, track_num):
        '''
        input:
            1. notes of one track of the music
            2. song name
        output:
            synthesized midi for both perf and score
        ''' 
        # create a PrettyMIDI file
        midi_file_perf = pretty_midi.PrettyMIDI()
        midi_file_score = pretty_midi.PrettyMIDI()

        # Create an instrument instance for a piano instrument
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano_perf = pretty_midi.Instrument(program=piano_program)
        piano_score = pretty_midi.Instrument(program=piano_program)

        for i in range(len(matrix)):
            pitch = int(matrix[i][3])
            start_t = matrix[i][5]
            end_t = matrix[i][7]
            velocity = int(matrix[i][4])
            note_perf = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_t, end=end_t)
            piano_perf.notes.append(note_perf)

            start_b = matrix[i][1]
            end_b = matrix[i][2]
            note_score = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_b, end=end_b)
            piano_score.notes.append(note_score)
        
        midi_file_perf.instruments.append(piano_perf)
        midi_file_score.instruments.append(piano_score)

        file_name_perf = song + "_perf.mid"
        midi_file_perf.write(self.perf_midi_dir + file_name_perf)    
        file_name_score = song + "_score.mid"
        midi_file_score.write(self.score_midi_dir + file_name_score)
    
    def get_track_index(self, notes):
        '''
        input:
            notes as numpy array
        output:
            [1, (track_num + 1)] 
            start indicies of each track with last row index included at last
        '''
        indices = []
        index = 0
        indices.append(0)
        for i in range(len(notes)):
            if notes[i, 0] != index:
                indices.append(i)
                index = notes[i, 0]
        return indices

    def pitch_1hot_encode(self, pitches, num_notes):
        classes = np.arange(0, 128)
        pitch_1hot = np.zeros((num_notes, 128))
        pitch = pitches.astype(int) - 1 
        pitch_1hot[np.arange(pitch.shape[0]), pitch] = 1
        return pitch_1hot.astype(int)

    def tempo_1hot_encode(self, tempos, num_notes):
        classes = np.arange(0, self.tempo_dim)
        tempo_1hot = np.zeros((num_notes, self.tempo_dim))
        tempos = tempos.astype(int) - 1
        c = np.where(tempos >= self.tempo_dim)
        tempos[c] = self.tempo_dim - 1
        tempo_1hot[np.arange(num_notes), tempos] = 1
        return tempo_1hot.astype(int)

    def durratio_1hot_encode(self, durratio, num_notes):
        classes = np.arange(0, self.durratio_dim)
        durratio_1hot = np.zeros((num_notes, self.durratio_dim))
        durratio = (durratio / self.durratio_base).astype(int)
        c = np.where(durratio >= self.durratio_dim)[0]
        # if c.shape[0] != 0:
        #     print(c.shape[0])
        #     raise ValueError
        durratio[c] = self.durratio_dim - 1
        durratio_1hot[np.arange(num_notes), durratio] = 1
        return durratio_1hot.astype(int)

    def dur_1hot_encode(self, dur, num_notes):
        dur = dur * 10 / 9;
        # print("dur", dur)
        classes = np.arange(0, self.dur_dim)
        dur_1hot = np.zeros((num_notes, self.dur_dim))
        # normalize duration by dividing base units
        # replace duration larger than 5s with 5s
        dur = (np.round(dur, decimals=4) / self.dur_base).astype(int)
        c = np.where(dur >= self.dur_dim)
        dur[c] = self.dur_dim - 1
        dur_1hot[np.arange(num_notes), dur] = 1
        return dur_1hot.astype(int)
        
    def dy_1hot_encode(self, dy, num_notes):
        classes = np.arange(0, self.dy_dim)
        dy_1hot = np.zeros((num_notes, self.dy_dim))
        dy_norm = (dy / 2).astype(int) - 1
        dy_1hot[np.arange(num_notes), dy_norm] = 1
        return dy_1hot.astype(int)

    def ioi_beat_encode(self, ioi_beat, num_notes):
        '''
        TODO: for now we replace the 1st ioi value in 2 bars with ioi_beat[0] % 8
        '''
        classes = np.arange(0, self.ioi_beat_dim)
        # print("ioi_beat", ioi_beat)
        ioi_beat_1hot = np.zeros((num_notes, self.ioi_beat_dim))
        c = np.where(ioi_beat >= self.max_dur_beat)[0]
        if (c.size > 0):
            # print(ioi_beat)
            ioi_beat[c] = ioi_beat[c] % self.max_dur_beat
            # print(ioi_beat)
        c = np.where(ioi_beat >= self.max_dur_beat)[0]
        assert(c.size == 0)

        ioi_beat_norm = (ioi_beat / self.ioi_beat_base).astype(int)
        # print(ioi_beat_norm) 
        # print((ioi_beat_norm) * self.ioi_beat_base)
        ioi_beat_1hot[np.arange(num_notes), ioi_beat_norm] = 1
        return ioi_beat_1hot.astype(int)
        
    
    def ioi_time_encode(self, ioi_total, num_notes):
        ioi = ioi_total[:, -1]
        ioi_beats = ioi_total[:, -2]
        ioi_start_b = ioi_total[:, 1]
        classes = np.arange(0, self.ioi_time_dim)
        # ioi[0] = 0 # TODO: we just set the first ioi as 0 in 2 bars
        c = np.where(ioi > 8)[0]
        if (c.size > 0):
            if (0 in c):
                ioi[0] = 0
                print(ioi)
                # print(ioi_total[:, 0])
                # print(ioi_start_b)

                # ioi -= ioi[0]
                # print(ioi)
            else:
                raise ValueError
        ioi_time_1hot = np.zeros((num_notes, self.ioi_time_dim))
        # print(ioi)
        ioi_time_norm = (ioi / self.ioi_time_base).astype(int)
        # print(ioi_time_norm)
        ioi_time_1hot[np.arange(num_notes), ioi_time_norm] = 1
        # print(np.where(ioi_time_1hot == 1))
        # raise ValueError
        return ioi_time_1hot

    def str_notes_to_1hot_txt(self, song_folder, score_dir, song_name):
        '''
        [archived]
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
            print(notes)
            notes_1hot[:, :, i] = self.str_notes_to_1hot_1song(notes, score)
            raise ValueError
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        np.save(parent_dir + '/data/pickle/' + song_name[:-4], notes_1hot) 


    def get_tempo_max_min(self, song_folder):
        '''
        incomplete code
        TODO: 
        1. fix code
        2. plot histogram

        '''
        for i in range(len(notes_2bars)):
            maxInColumns = np.amax(notes_2bars[i], axis=0)
            minInColumns = np.amin(notes_2bars[i], axis=0)
            if (maxInColumns[5] > tempo_max):
                tempo_max = maxInColumns[5]
            if (maxInColumns[7] > tempo_max):
                tempo_max = maxInColumns[7]
            if (minInColumns[5] < tempo_min):
                tempo_min = minInColumns[5]
            if (minInColumns[7] < tempo_min):
                tempo_min = minInColumns[7]
        print("MAX Tempo:", tempo_max)
        print("MIN Tempo:", tempo_min)