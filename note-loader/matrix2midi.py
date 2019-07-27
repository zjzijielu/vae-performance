import pretty_midi
import numpy as np
from os import listdir
from os.path import isfile, join
from utils import *
from params import parameters

'''
synthesize 2 bar music
'''

p = parameters()
# dimensions
score_1hot_dim = p.score_1hot_dim
perf_1hot_dim = p.perf_1hot_dim
durratio_dim = p.durratio_dim
dy_dim = p.dy_dim
ioi_time_dim = p.ioi_time_dim

# score idx
pitch_idx = p.pitch_idx
ioi_beat_idx = p.ioi_beat_idx
dur_idx = p.dur_idx

# perf idx
durratio_idx = p.durratio_idx - score_1hot_dim
dy_idx = p.dy_idx - score_1hot_dim
ioi_time_idx = p.ioi_time_idx - score_1hot_dim

# base 
dur_base = p.dur_base
durratio_base = p.durratio_base

def synthesize(recon_x, folder, song_name):
    # pitch 
    pitch = np.where(recon_x[:, pitch_idx:ioi_beat_idx] == 1)[1] + 1

    # dynamic
    dy = np.where(recon_x[:, dy_idx:ioi_time_idx] == 1)[1] + 1

    # ioi
    ioi_time = np.where(recon_x[:, ioi_time_idx:])
    ioi_time = ioi_time * ioi_time_base
    
    # duration
    dur_beats = np.where(recon_x[:, dur_idx:durratio_idx] == 1)[1]
    dur_beats = dur_beats * dur_beats
    durratio = np.where(recon_x[:, durratio_idx:dy_idx] == 1)[1]
    durratio = durratio * durratio_base
    perf_dur = dur_beats * durratio

    midi_file = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano_perf = pretty_midi.Instrument(program=piano_program)

    start_t = 0
    for i in range(len(m)):
        p = pitch[i]
        start_t = start_t + ioi_time[i]
        end_t = start_t + perf_dur[i]
        velocity = dy[i]
        note_perf = pretty_midi.Note(velocity=velocity, pitch=p, start=start_t, end=end_t)
        piano_perf.notes.append(note_perf)

    midi_file.instruments.append(piano_perf)
    file_name_perf = song_name + ".mid" 
    midi_file.write(folder + file_name_perf)
    