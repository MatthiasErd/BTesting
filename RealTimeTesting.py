from essentia.streaming import *
from essentia import Pool, run, array, reset
from time import time
import numpy as np
import soundcard as sc
from struct import unpack

# Mic-Input
print(sc.default_microphone())

# protobuf file
modelName = '/home/maedd/Documents/ADT_training/adt-training/data/experiments/1604938592spec/musicnn_drums_librosa.pb'
input_layer = 'model/Placeholder'
output_layer = 'model/Sigmoid'

# automatic drum transcription labels
adt_labels = ['snare','kick']
nLabels = len(adt_labels)

sampleRate = 16000
frameSize=512
hopSize=128
numberBands = 96

# model parameters
patchSize = 3

# audio buffers to record to for prediction and peak pipeline
bufferSize = patchSize * hopSize
buffer = np.zeros(bufferSize, dtype='float32')
bufferPeak = np.zeros(256, dtype='float32')

# Algorithms for prediction
vimp = VectorInput(buffer)
fc = FrameCutter(frameSize=frameSize, hopSize=hopSize)
w = Windowing(normalized=False)
spec = Spectrum()
mel = MelBands(numberBands=numberBands, sampleRate=sampleRate,
               highFrequencyBound=sampleRate // 2,
               inputSize=frameSize,
               weighting='linear', normalize='unit_tri',
               warpingFormula='slaneyMel')
vtt = VectorRealToTensor(shape=[1, 1, patchSize, numberBands], lastPatchMode='discard')
ttp = TensorToPool(namespace=input_layer)
tfp = TensorflowPredict(graphFilename=modelName,
                        inputs=[input_layer],
                        outputs=[output_layer], isTrainingName='model/Placeholder_2')
ptt = PoolToTensor(namespace=output_layer)
ttv = TensorToVectorReal()
pool = Pool()

# Algorithms for peak picking
input = VectorInput(bufferPeak)
fc2 = FrameCutter(frameSize=256, hopSize=256)
pk = PeakDetection()
m = Mean()
rms = RMS()

# prediction pipeline
vimp.data      >> fc.signal
fc.frame       >>  w.frame
w.frame        >>  spec.frame
spec.spectrum  >>  mel.spectrum
mel.bands      >>  (pool, "melbands")
mel.bands      >>  vtt.frame
vtt.tensor     >>  ttp.tensor
ttp.pool       >>  tfp.poolIn
tfp.poolOut    >>  ptt.pool
ptt.tensor     >>  ttv.tensor
ttv.frame      >>  (pool, output_layer)

# peak picking pipeline
input.data      >>  fc2.signal
fc2.frame        >> pk.array
pk.positions    >> None
pk.amplitudes   >> rms.array
rms.rms         >> (pool, 'peaks')


def callback(data):

    # update audio buffer
    buffer[:] = array(unpack('f' * bufferSize, data))

    # generate predictions
    reset(vimp)
    run(vimp)

    return pool[output_layer]

def peakPicking(data):

    # update audio buffer
    bufferPeak[:] = array(unpack('f' * 256, data))

    # generate predictions
    reset(input)
    run(input)

    # calculate mean
    peaks = np.array(pool['peaks'])
    mean = np.mean(peaks[:-1])

    pool.clear()

    return mean


pool.clear()

# peak threshold
val = 0.03

with sc.all_microphones(include_loopback=True)[0].recorder(samplerate=sampleRate) as mic:
    time_thresh = time()
    while True:

        # run peak detection
        peak = peakPicking(mic.record(numframes=256).mean(axis=1))

        # record buffer for possible prediction
        data = mic.record(numframes=bufferSize).mean(axis=1)

        if peak > val and time() - time_thresh > 0.3:

            # reset time_thresh
            time_thresh = time()

            # start time to measure prediction
            start_time = time()

            # run prediction
            callback(data)

            # sort and print predictions
            list = pool[output_layer][-1, :].T.tolist()
            list = np.array(list)

            for i, l in enumerate(list.argsort()[-2:][::-1]):
                print('{}: {}'.format(i, adt_labels[l]))
                print(pool[output_layer])

            # print the time that it took to run a prediction
            print('Prediction time: {:.2f}s \n\n'.format(time() - start_time))

