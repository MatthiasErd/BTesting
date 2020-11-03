from essentia.standard import *
import tensorflow as tf
import numpy as np

# disabling deprecation warnings (caused by change from tensorflow 1.x to 2.x)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


SAVED_MODEL_PATH = "/media/maedd/USB DISK/Backup/10_30_2020/Files/h5/EssentiaStandard.h5"
SAMPLES_TO_CONSIDER = 1024
sampleRate = 16000
frameSize=512
hopSize=256
numberBands = 96
weighting='linear'
warpingFormula='slaneyMel'
normalize='unit_tri'

# instantiate the essentia analysing algorithms
w = Windowing(normalized=False)
spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
mfcc = MelBands(numberBands=numberBands, sampleRate=sampleRate,
               highFrequencyBound=sampleRate // 2,
               inputSize=frameSize // 2 + 1,
               weighting=weighting, normalize=normalize,
               warpingFormula=warpingFormula)

MFCCs = []

# disabling deprecation warnings (caused by change from tensorflow 1.x to 2.x)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class _Keyword_Spotting_Service:

    model = None
    _mapping = [
        "snare",
        "kick"
    ]
    _instance = None


    def predict(self, file_path):

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        MFCCs = np.array(MFCCs)

        print(MFCCs.shape)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        print(MFCCs.shape)

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=512, hop_length=256):

        # load audio file
        loader = MonoLoader(filename=file_path, sampleRate=sampleRate)
        audio = loader()

        if (len(audio) > SAMPLES_TO_CONSIDER):
            for frame in FrameGenerator(audio[:SAMPLES_TO_CONSIDER], frameSize=frameSize, hopSize=hopSize,
                                            startFromZero=True):
                mfcc_bands = mfcc(spectrum(w(frame)))  # can not set argument numberBands in mfcc (?)
                MFCCs.append(essentia.array(mfcc_bands.tolist()))
        else:
            print("Sample to small")
            print("{}: {}".format(file_path, i - 1))
        #print(MFCCs)
        return MFCCs

def Keyword_Spotting_Service():

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance




if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1

    # make a prediction
    keyword = kss.predict("/home/maedd/Documents/Sounds/Kick/DR-660Kick31.wav")
    print(keyword)