from essentia.standard import *
import os
import numpy as np
import json

DATASET_PATH = "/home/maedd/Documents/Sounds"
JSON_PATH = "/home/maedd/Documents/Bachelorarbeit/Network/Files/dataEssentiaStandardTryOut.json"

numSamplesToCompute = 1024
sampleRate = 16000
frameSize=512
hopSize=256
numberBands = 96

mel = []

# instantiate the essentia analysing algorithms
w = Windowing(normalized=False)
spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
melBands = MelBands(numberBands=numberBands, sampleRate=sampleRate,
               highFrequencyBound=sampleRate // 2,
               inputSize=frameSize,
               weighting='linear', normalize='unit_tri',
               warpingFormula='slaneyMel')

def preprocess_dataset(dataset_path, json_path, num_mfcc=40, n_fft=512, hop_length=64):

    # dictionary
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file
                loader = MonoLoader(filename=file_path, sampleRate=sampleRate)
                audio = loader()

                # compute melBands
                if (len(audio) > 1024):
                    for frame in FrameGenerator(audio[:numSamplesToCompute], frameSize=frameSize, hopSize=hopSize, startFromZero=True):

                        mel_bands = melBands(spectrum(w(frame)))
                        mel.append(mel_bands)
                else:
                    print("Sample to small")
                    print("{}: {}".format(file_path, i - 1))

                # store data into array
                data["MFCCs"].append(essentia.array(mel).tolist())
                data["labels"].append(i-1)
                data["files"].append(file_path)
                print("{}: {}".format(file_path, i-1))
                #test = np.array(data["MFCCs"])
                #print(test.shape)
                mel.clear()

    # save data to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":

   preprocess_dataset(DATASET_PATH, JSON_PATH)
   print("done")