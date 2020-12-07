from essentia.streaming import *
from essentia import Pool, run
from time import time
import numpy as np

# musicnn_drums_librosa.pb pathcsize = 5 works, tendiert aber mehr zu snare predictions

modelName = '/home/maedd/Documents/BThesis/data/drums[1024].pb'
#modelName = '/home/maedd/Desktop/Finals/audio/12_01_2020/drums_Librosa.pb'
input_layer = 'model/Placeholder'
output_layer = 'model/Sigmoid'


msd_labels = ['snare','hihat', 'kick']

weighting='linear'
warpingFormula='slaneyMel'
normalize='unit_tri'

sampleRate = 16000
frameSize=512
hopSize=256
numberBands = 96

# model parameters
patchSize = 2

filename = '/home/maedd/Music/Experiments/FirstDrumSamples/Snare/Miscellaneous Snare (6196).wav'

# Algorithms for mel-spectrogram computation
audio = MonoLoader(filename=filename, sampleRate=sampleRate)

fc = FrameCutter(frameSize=frameSize, hopSize=hopSize)

w = Windowing(normalized=False)

spec = Spectrum()

mel = MelBands(numberBands=numberBands, sampleRate=sampleRate,
               highFrequencyBound=sampleRate // 2,
               inputSize=frameSize // 2 + 1,
               weighting=weighting, normalize=normalize,
               warpingFormula=warpingFormula)

# Algorithms for logarithmic compression of mel-spectrograms
shift = UnaryOperator(shift=1, scale=10000)

comp = UnaryOperator(type='log10')

# This algorithm cuts the mel-spectrograms into patches
# according to the model's input size and stores them in a data
# type compatible with TensorFlow
vtt = VectorRealToTensor(shape=[1, 1, patchSize, numberBands])

# Auxiliar algorithm to store tensors into pools
ttp = TensorToPool(namespace=input_layer)

# The core TensorFlow wrapper algorithm operates on pools
# to accept a variable number of inputs and outputs
tfp = TensorflowPredict(graphFilename=modelName,
                        inputs=[input_layer],
                        outputs=[output_layer], isTrainingName="model/Placeholder_2") # isTrainingName="model/Placeholder_2"

# Algorithms to retrieve the predictions from the wrapper
ptt = PoolToTensor(namespace=output_layer)

ttv = TensorToVectorReal()

# Another pool to store output predictions
pool = Pool()


audio.audio    >>  fc.signal
fc.frame       >>  w.frame
w.frame        >>  spec.frame
spec.spectrum  >>  mel.spectrum
mel.bands      >>  shift.array
shift.array    >>  comp.array
comp.array     >>  vtt.frame
vtt.tensor     >>  ttp.tensor
ttp.pool       >>  tfp.poolIn
tfp.poolOut    >>  ptt.pool
ptt.tensor     >>  ttv.tensor
ttv.frame      >>  (pool, output_layer)

# Store mel-spectrograms to reuse them later in this tutorial
comp.array     >>  (pool, "melbands")


# prediction time
start_time = time()

#pool.clear()

# run algorithms
run(audio)

print(pool[output_layer])
#print(pool[output_layer][-1, :].T.tolist())



print('Most predominant tags:')

list = []

test = pool[output_layer].T.tolist()

snare = np.mean(test[0])
hihat = np.mean(test[1])
kick = np.mean(test[2])

print(snare)
print(hihat)
print(kick)

list.append(snare)
list.append(hihat)
list.append(kick)

prediction = list.index(max(list))


print(msd_labels[prediction])

print('Prediction time: {:.2f}s'.format(time() - start_time))










