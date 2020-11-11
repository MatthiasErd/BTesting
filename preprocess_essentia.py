#Copyright (c) Music Technology Group, Universitat Pompeu Fabra 2019. Code developed by Jordi Pons (www.jordipons.me)

from essentia.standard import *
import os
import librosa
from joblib import Parallel, delayed
import json
import config_file
import argparse
import pickle
import numpy as np
from pathlib import Path

DEBUG = True

# instantiate the essentia analysing algorithms
mel = []

w = Windowing(normalized=False)
spectrum = Spectrum(size=1024)  # FFT() would return the complex FFT, here we just want the magnitude spectrum
melBands = MelBands(numberBands=96, sampleRate=16000,
               highFrequencyBound=16000 // 2,
               inputSize=512,
               weighting='linear', normalize='unit_tri',
               warpingFormula='slaneyMel')

def compute_audio_repr(audio_file, audio_repr_file):

    # load with essentia
    loader = MonoLoader(filename=audio_file, sampleRate=16000)
    audio = loader()

    if config['type'] == 'waveform':
        audio_repr = audio
        audio_repr = np.expand_dims(audio_repr, axis=1)

    elif config['spectrogram_type'] == 'mel':

        if (len(audio) > 1024):
            for frame in FrameGenerator(audio, frameSize=config['n_fft'], hopSize=config['hop'],
                                        startFromZero=True):
                mel_bands = melBands(spectrum(w(frame)))
                mel.append(mel_bands)
            audio_repr = essentia.array(mel)
            mel.clear()

        else:
            print("sample to small", audio_file)

    # Compute length
    print(audio_repr.shape)
    length = audio_repr.shape[0]

    # Transform to float16 (to save storage, and works the same)
    audio_repr = audio_repr.astype(np.float16)

    # Write results:
    with open(audio_repr_file, "wb") as f:
        pickle.dump(audio_repr, f)  # audio_repr shape: NxM

    return length


def do_process(files, index):

    try:
        [id, audio_file, audio_repr_file] = files[index]
        if not os.path.exists(audio_repr_file[:audio_repr_file.rfind('/') + 1]):
            path = Path(audio_repr_file[:audio_repr_file.rfind('/') + 1])
            path.mkdir(parents=True, exist_ok=True)
        # compute audio representation (pre-processing)
        length = compute_audio_repr(audio_file, audio_repr_file)
        # index.tsv writing
        fw = open(config_file.DATA_FOLDER + config['audio_representation_folder'] + "index_" + str(config['machine_i']) + ".tsv", "a")
        fw.write("%s\t%s\t%s\n" % (id, audio_repr_file[len(config_file.DATA_FOLDER):], audio_file))
        fw.close()
        print(str(index) + '/' + str(len(files)) + ' Computed: %s' % audio_file)

    except Exception as e:
        ferrors = open(config_file.DATA_FOLDER + config['audio_representation_folder'] + "errors" + str(config['machine_i']) + ".txt", "a")
        ferrors.write(audio_file + "\n")
        ferrors.write(str(e))
        ferrors.close()
        print('Error computing audio representation: ', audio_file)
        print(str(e))


def process_files(files):

    if DEBUG:
        print('WARNING: Parallelization is not used!')
        for index in range(0, len(files)):
            do_process(files, index)

    else:
        Parallel(n_jobs=config['num_processing_units'])(
            delayed(do_process)(files, index) for index in range(0, len(files)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('configurationID', help='ID of the configuration dictionary')
    args = parser.parse_args()
    config = config_file.config_preprocess[args.configurationID]

    config['audio_representation_folder'] = "audio_representation/%s__%s/" % (config['identifier'], config['type'])
    # set audio representations folder
    if not os.path.exists(config_file.DATA_FOLDER + config['audio_representation_folder']):
        os.makedirs(config_file.DATA_FOLDER + config['audio_representation_folder'])
    else:
        print("WARNING: already exists a folder with this name!"
              "\nThis is expected if you are splitting computations into different machines.."
              "\n..because all these machines are writing to this folder. Otherwise, check your config_file!")

    # list audios to process: according to 'index_file'
    files_to_convert = []
    f = open(config_file.DATA_FOLDER + config["index_file"])
    for line in f.readlines():
        id, audio = line.strip().split("\t")
        audio_repr = audio[:audio.rfind(".")] + ".pk" # .npy or .pk
        files_to_convert.append((id, config['audio_folder'] + audio,
                                 config_file.DATA_FOLDER + config['audio_representation_folder'] + audio_repr))

    # compute audio representation
    if config['machine_i'] == config['n_machines'] - 1:
        process_files(files_to_convert[int(len(files_to_convert) / config['n_machines']) * (config['machine_i']):])
        # we just save parameters once! In the last thread run by n_machine-1!
        json.dump(config, open(config_file.DATA_FOLDER + config['audio_representation_folder'] + "config.json", "w"))
    else:
        first_index = int(len(files_to_convert) / config['n_machines']) * (config['machine_i'])
        second_index = int(len(files_to_convert) / config['n_machines']) * (config['machine_i'] + 1)
        assigned_files = files_to_convert[first_index:second_index]
        process_files(assigned_files)

    print("Audio representation folder: " + config_file.DATA_FOLDER + config['audio_representation_folder'])