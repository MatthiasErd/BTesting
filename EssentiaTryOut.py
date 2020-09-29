import numpy as np
import matplotlib.pyplot as plt
import soundcard as sc

from struct import unpack
from IPython import display

from essentia.streaming import *
from essentia import Pool, run, array, reset

from scipy.special import softmax

# model parameters
modelName = '/home/maedd/Documents/Bachelorarbeit/Network/xor.pb'
inputLayer = 'conv2d_input'
outputLayer = 'training/Adam/dense_1/bias/v'
labels = ['kick', 'snare']
nLabels = len(labels)

sampleRate = 16000
frameSize = 512
hopSize = 64
numberBands = 13

# analysis parameters
patchSize = 64
displaySize = 10

bufferSize = patchSize * hopSize
#
buffer = np.zeros(bufferSize, dtype='float32')

vimp = VectorInput(buffer)

fc = FrameCutter(frameSize=frameSize, hopSize=hopSize)

tim = TensorflowInputMusiCNN()

vtt = VectorRealToTensor(shape=[1, 1, patchSize, numberBands],
                         lastPatchMode='discard')

ttp = TensorToPool(namespace=inputLayer)

tfp = TensorflowPredict(graphFilename=modelName,
                        inputs=[inputLayer],
                        outputs=[outputLayer])

ptt = PoolToTensor(namespace=outputLayer)

ttv = TensorToVectorReal()

spec = Spectrum()
mfcc = MFCC()

pool = Pool()




vimp.data >> fc.signal
fc.frame >> spec.frame
spec.spectrum >> mfcc.spectrum
mfcc.bands >> None
mfcc.mfcc >> (pool, 'lowlevel.mfcc')
mfcc.mfcc >> vtt.frame
vtt.tensor >> ttp.tensor
ttp.pool >> tfp.poolIn
tfp.poolOut >> ptt.pool
ptt.tensor >> ttv.tensor
ttv.frame >> (pool, outputLayer)



def callback(data):
    # update audio buffer
    buffer[:] = array(unpack('f' * bufferSize, data))

    # generate predictions
    reset(vimp)
    run(vimp)

    # update mel and activation buffers
    mfccBuffer[:] = np.roll(mfccBuffer, -patchSize)
    mfccBuffer[:, -patchSize:] = pool['lowlevel.mfcc'][-patchSize:, :].T

    actBuffer[:] = np.roll(actBuffer, -1)
    actBuffer[:, -1] = softmax(20 * pool['training/Adam/dense_1/bias/v'][-1, :].T)

    print(pool['training/Adam/dense_1/bias/v'])
    #print(mfccBuffer)

    pool.clear()

    # update plots
    f.canvas.draw()


# initialize plot buffers
mfccBuffer = np.zeros([numberBands, patchSize * displaySize])
actBuffer = np.zeros([nLabels, displaySize])


# initialize plots
f, ax = plt.subplots(1, 2, figsize=[9.6, 7])
f.canvas.draw()



# capture and process the speakers loopback
with sc.all_microphones(include_loopback=True)[0].recorder(samplerate=sampleRate) as mic:
    while True:
        callback(mic.record(numframes=bufferSize).mean(axis=1))


