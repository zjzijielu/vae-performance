import pretty_midi
import numpy as np
from os import listdir
from os.path import isfile, join
from utils import *

'''
check if 2 bars data is correct
'''

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_folder = parent_dir + "/data/2bars_data/"
output_dir = parent_dir + "/data/perf_midi_syn/"

# score part
pitch_dim = 128
ioi_beat_dim = 128 # base 1/16, longest 8 beats
dur_dim = 64 # base 1/16 beat, longest 4 beats

# performance part
durratio_dim = 250 # base is 0.1, from 0 to 10
dy_dim = 64
ioi_time_dim = 801 # base is 0.01s, maximum 8s

# total dimension
score_1hot_dim = pitch_dim + ioi_beat_dim + dur_dim
perf_1hot_dim = durratio_dim + dy_dim + ioi_time_dim
onehot_dim = score_1hot_dim + perf_1hot_dim

# idx of each 1hot vector
pitch_idx = 0
ioi_beat_idx = pitch_dim
dur_idx = ioi_beat_idx + ioi_beat_dim

durratio_idx = score_1hot_dim
dy_idx = durratio_idx + durratio_dim
ioi_time_idx = dy_idx + dy_dim

# base values
dur_base = 0.0625 # 1/16 beat
ioi_beat_base = 0.0625
durratio_base = 0.02
ioi_time_base = 0.01
max_dur_beat = 8
min_notes_num = 3
dy_base = 2


# get files
files = find('*.npy', data_folder)
for i, f_name in enumerate(files):
    print(f_name)
    m = np.load(data_folder + f_name)

    # pitch
    pitch = np.where(m[:, pitch_idx:ioi_beat_idx] == 1)[1] + 1
    print(pitch)
    
    # dynamic
    dy = np.where(m[:, dy_idx:ioi_time_idx] == 1)[1]
    print("dy:", dy)

    # ioi
    ioi_time = np.where(m[:, ioi_time_idx:] == 1)[1]
    ioi_time = ioi_time * ioi_time_base
    print("ioi:", ioi_time)

    # ioi beats
    ioi_beat = np.where(m[:, ioi_beat_idx:dur_idx] == 1)[1]
    print("ioi_beat idx", ioi_beat)
    ioi_beat = ioi_beat * ioi_beat_base
    print("ioi beats:", ioi_beat)
    # duration
    dur_beats = np.where(m[:, dur_idx:durratio_idx] == 1)[1]
    print("dur beats idx:", dur_beats)
    dur_beats = dur_beats * dur_base
    durratio = np.where(m[:, durratio_idx:dy_idx] == 1)[1]
    durratio = durratio * durratio_base
    print("dur_beats:", dur_beats)
    print(durratio)
    print(dur_beats * durratio)
    perf_dur = dur_beats * durratio
    print("perf_dur:", perf_dur)

    # synthesize midi file 
    midi_file_perf = pretty_midi.PrettyMIDI()
    midi_file_score = pretty_midi.PrettyMIDI()
    # Create an instrument instance for a piano instrument
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano_perf = pretty_midi.Instrument(program=piano_program)
    piano_score = pretty_midi.Instrument(program=piano_program)

    start_t = 0
    for i in range(len(m)):
        p = pitch[i]
        start_t = start_t + ioi_time[i]
        end_t = start_t + perf_dur[i]
        velocity = dy[i] * dy_base
        note_perf = pretty_midi.Note(velocity=velocity, pitch=p, start=start_t, end=end_t)
        piano_perf.notes.append(note_perf)

    start_t = 0
    for i in range(len(m)):
        p = pitch[i]
        start_t = start_t + ioi_beat[i]
        end_t = start_t + dur_beats[i]
        note_score = pretty_midi.Note(velocity=80, pitch=p, start=start_t, end=end_t)
        piano_score.notes.append(note_score) 

    midi_file_perf.instruments.append(piano_perf)
    file_name_perf = f_name[:-4] + ".mid"
    midi_file_perf.write(output_dir + file_name_perf)

    midi_file_score.instruments.append(piano_score)
    file_name_score = f_name[:-4] + "_score.mid"
    midi_file_score.write(output_dir + file_name_score)

    # check if tempo has changes
    midi_data = pretty_midi.PrettyMIDI(output_dir + file_name_perf)
    tempi = midi_data.get_tempo_changes()
    print("#" * 5, "tempi", "#" * 5)
    print(tempi)
    if i == 3:
        raise ValueError
    

# synthesize 2 bar midi file
