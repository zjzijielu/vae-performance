import argparse
from data_loader import MidiLoader
from utils import *

song = 'boy_acc'
aligned = True
data_folder = '../data/'

def main():
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
    loader.str_notes_to_1hot(song_folder, score_dir)

if __name__ == '__main__':
    main()