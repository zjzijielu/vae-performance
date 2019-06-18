import argparse
from data_loader import MidiLoader
from utils import *

def main_txt():
    '''
    [archived] this main function is used to load txt data of aligned data
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--song_name', '-s', type=str, default='boy_acc',
                        help="which song to load")
    parser.add_argument('--aligned', '-a', type=int, default=1,
                        help='if aligned')
    
    args = parser.parse_args()
    song = args.song_name
    aligned = args.aligned
    # get parent directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    song_folder = parent_dir + "/data/"
    '''
    the aligned fgt folder is "data/aligned_fgt/"
    the raw folder is "data/raw/"
    '''
    if aligned == 1:
        song_folder += 'aligned_fgt/fgt_' + song + '/'
    else:
        song_folder += 'raw/' + song + '/'
    
    # user midiloader to convert all txt to 1hot vectors
    score_dir = song_folder + 'score/'
    loader = MidiLoader()
    loader.str_notes_to_1hot(song_folder, score_dir, song)

def main():
    '''
    load the data houze, which is raw performance data in csv format
    we convert these data into 2 bar performance data and store in ../data/performance/2bar_data/
    '''
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    song_folder = parent_dir + "/data/perf_tempo_csv/"
    
    loader = MidiLoader()
    loader.str_notes_to_1hot(song_folder)
    

    


if __name__ == '__main__':
    main()