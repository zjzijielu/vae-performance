import pretty_midi
import os

pitches = [61, 64, 61, 57, 59, 61, 62, 73, 74, 73]
start_t = [48, 48, 48.8, 49.06667, 49.3333, 49.6, 50.312047, 50.537134, 50.858487, 51.207091]
end_t = [48.72, 48.72, 49.04, 49.306667, 49.573333, 50.312047, 50.951363, 50.825154, 51.170539, 51.549137]

midi_file_perf = pretty_midi.PrettyMIDI()
piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano_perf = pretty_midi.Instrument(program=piano_program)

for i in range(len(pitches)):
    pitch = pitches[i]
    st = start_t[i] - 48
    et = end_t[i] - 48
    note_perf = pretty_midi.Note(velocity=80, pitch=pitch, start=st, end=et)
    piano_perf.notes.append(note_perf)

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_folder = parent_dir + "/data/2bars_data/"
output_dir = parent_dir + "/data/perf_midi_syn/"

midi_file_perf.instruments.append(piano_perf)
midi_file_perf.write(output_dir + "test.mid")

midi_data = pretty_midi.PrettyMIDI(output_dir + "test.mid")
tempi = midi_data.get_tempo_changes()
print("#" * 5, "tempi", "#" * 5)
print(tempi)
