from essentia.standard import *
from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
import numpy as np
import librosa

file_path = '/home/maedd/Documents/Sounds/Kick/RY30Kick01.wav'

numSamplesToCompute = 1024
sampleRate = 16000
frameSize=512
hopSize=256
numberBands = 96

mel_essentia = []
mel_librosa = []

# instantiate the essentia analysing algorithms
w = Windowing(normalized=False)
spectrum = Spectrum(size=numSamplesToCompute)  # FFT() would return the complex FFT, here we just want the magnitude spectrum
melBands = MelBands(numberBands=numberBands, sampleRate=sampleRate,
               highFrequencyBound=sampleRate // 2,
               inputSize=frameSize,
               weighting='linear', normalize='unit_tri',
               warpingFormula='slaneyMel')

# load audio file
loader = MonoLoader(filename=file_path, sampleRate=sampleRate)
audio = loader()

# compute Essentia melBands
if (len(audio) > 1024):
    for frame in FrameGenerator(audio[:numSamplesToCompute], frameSize=frameSize, hopSize=hopSize, startFromZero=True):
        mel_bands = melBands(spectrum(w(frame)))
        mel_essentia.append(mel_bands)
    mel_essentia = essentia.array(mel_essentia)
    mel_essentia = mel_essentia.astype(np.float16)

# computing librosa melBands
mel_bands2 = librosa.feature.melspectrogram(audio[:numSamplesToCompute], sr=sampleRate, hop_length=hopSize, n_fft=frameSize, n_mels=numberBands).T
mel_librosa = mel_bands2.astype(np.float16)

# print results of essentia/librosa computation
print(mel_essentia.shape)  # prints: "(9, 96)"
print(np.mean(mel_essentia)) # prints: "3.219"

print(mel_librosa.shape)  # prints: "(10, 96)"
print(np.mean(mel_librosa)) #prints: "3.688"

















