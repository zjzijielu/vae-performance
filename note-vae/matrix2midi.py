import pretty_midi
import numpy as np
from os import listdir
from os.path import isfile, join
from utils import *
from params import parameters
import torch

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
ioi_time_base = p.ioi_time_base

def synthesize(recon_x, score, target, folder, song_name):
    print(recon_x.size())
    # pitch 
    pitch = np.where(score[:, pitch_idx:ioi_beat_idx] == 1)[1] + 1

    # dynamic
    dy = torch.argmax(recon_x[:, dy_idx:ioi_time_idx], dim=1)
    print("dy:", dy)
    dy_gt = torch.argmax(target[:, dy_idx:ioi_time_idx], dim=1)
    print("dy_gt:", dy_gt)
    
    # ioi
    ioi_time = torch.argmax(recon_x[:, ioi_time_idx:], dim=1)
    ioi_time_gt = torch.argmax(target[:, ioi_time_idx:], dim=1)
    print("ioi_time:", ioi_time)
    print("ioi_time_gt:", ioi_time_gt)
    print(recon_x[:, ioi_time_idx:])
    # ioi_time = np.where(recon_x[:, ioi_time_idx:] == 1)[1]
    ioi_time = ioi_time * ioi_time_base
    
    # duration
    dur_beats = np.where(score[:, dur_idx:] == 1)[1]
    dur_beats = dur_beats * dur_base
    durratio = torch.argmax(recon_x[:, durratio_idx:dy_idx], dim=1)
    durratio_gt = torch.argmax(target[:, durratio_idx:dy_idx], dim=1)
    print("durratio:", durratio)
    print("durratio_gt:", durratio_gt)
    durratio = durratio * durratio_base
    perf_dur = dur_beats * durratio
    print(dur_beats)

    midi_file = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano_perf = pretty_midi.Instrument(program=piano_program)

    start_t = 0
    for i in range(len(recon_x)):
        p = int(pitch[i])
        start_t = start_t + ioi_time[i]
        end_t = start_t + perf_dur[i].long()
        velocity = int(dy[i])
        note_perf = pretty_midi.Note(velocity=velocity, pitch=p, start=start_t, end=end_t)
        piano_perf.notes.append(note_perf)

    midi_file.instruments.append(piano_perf)
    file_name_perf = song_name + ".mid" 
    print(folder)
    print(file_name_perf)
    midi_file.write(folder + file_name_perf)
    