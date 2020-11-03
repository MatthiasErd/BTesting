from essentia.streaming import *
from essentia import Pool, run, array, reset
from time import time
import numpy as np
import matplotlib.pyplot as plt
import soundcard as sc
from struct import unpack

# works with the msd-musicnn.pb file from Essentia homepage
# --> (patchSize: 187, inputLayer = 'model/Placeholder', output_layer = 'model/Sigmoid'


# kerasModel_essentia.pb compiles with no errors, but predictions are always the same (e.g. [[0.0023], [0.0023]] (mel Bands in preprocess computed with essentia)
# (exported to .pb NOT with Alonsos keras-onnx-tensorflow repository)
# --> input_layer = 'conv2d_input', output_layer = 'training/Adam/dense_1/bias/v'


# does not work with kerasModel_AlonsoConverter.pb
# (exported to .pb with keras-onnx-tensorflow repository)
# --> input_layer = 'dense_1/kernel_tf_0_ab98e656', output_layer = 'training/Adam/dense_1/bias/v'
#  File "/home/maedd/.local/lib/python3.6/site-packages/essentia/__init__.py", line 148, in run
#    return _essentia.run(gen)
# RuntimeError: TensorflowPredict: Error running the Tensorflow session. You must feed a value for placeholder tensor 'conv2d_input' with dtype float and shape [?,3,96,1]
#	 [[{{node conv2d_input}}]]


# does not work with 'musicnn_drums_essentia.pb' (computetd with musicnn-training repository, preprocess with essentia)
# patchSize: 3, input_layer = 'model/Placeholder', output_layer = 'gradients/model/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad'
#   File "/home/maedd/.local/lib/python3.6/site-packages/essentia/__init__.py", line 148, in run
#     return _essentia.run(gen)
# RuntimeError: TensorflowPredict: Error running the Tensorflow session. You must feed a value for placeholder tensor 'model/Placeholder_1' with dtype float and shape [?,2]
# 	 [[{{node model/Placeholder_1}}]]


# does not work with 'musicnn_drums_librosa.pb' (computetd with musicnn-training repository, preprocess with librosa)
# patchSize: 3, input_layer = 'model/Placeholder', output_layer = 'gradients/model/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad'
#   File "/home/maedd/.local/lib/python3.6/site-packages/essentia/__init__.py", line 148, in run
#     return _essentia.run(gen)
# RuntimeError: TensorflowPredict: Error running the Tensorflow session. You must feed a value for placeholder tensor 'model/Placeholder_1' with dtype float and shape [?,2]
# 	 [[{{node model/Placeholder_1}}]]



modelName = '/media/maedd/USB DISK/Backup/10_30_2020/Files/Protobufs/musicnn_drums_essentia.pb'
input_layer = 'model/Placeholder'
output_layer = 'gradients/model/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad'

msd_labels = ['snare','kick']
nLabels = len(msd_labels)

SAMPLES_TO_CONSIDER = 1024
sampleRate = 16000
frameSize=512
hopSize=256
numberBands = 96
weighting='linear'
warpingFormula='slaneyMel'
normalize='unit_tri'

# model parameters
patchSize = 3
displaySize = 10

bufferSize = patchSize * hopSize
buffer = np.zeros(bufferSize, dtype='float32')

# input sample for prediction
filename = '/home/maedd/Documents/Sounds/Kick/Kick1.wav'

# Algorithms for mel-spectrogram computation
vimp = VectorInput(buffer)

fc = FrameCutter(frameSize=frameSize, hopSize=hopSize)

w = Windowing(normalized=False)

spec = Spectrum()

mel = MelBands(numberBands=numberBands, sampleRate=sampleRate,
               highFrequencyBound=sampleRate // 2,
               inputSize=frameSize,
               weighting=weighting, normalize=normalize,
               warpingFormula=warpingFormula)

vtt = VectorRealToTensor(shape=[1, 1, patchSize, numberBands], lastPatchMode='discard')

ttp = TensorToPool(namespace=input_layer)

tfp = TensorflowPredict(graphFilename=modelName,
                        inputs=[input_layer],
                        outputs=[output_layer])

ptt = PoolToTensor(namespace=output_layer)

ttv = TensorToVectorReal()

pool = Pool()

# connect the algorithms
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


def callback(data):

    # update audio buffer
    buffer[:] = array(unpack('f' * bufferSize, data))

    # generate predictions
    reset(vimp)
    run(vimp)

    # print predictions
    print(pool[output_layer])


pool.clear()


# capture and process the speakers loopback
with sc.all_microphones(include_loopback=True)[0].recorder(samplerate=sampleRate) as mic:
    while True:
        callback(mic.record(numframes=bufferSize).mean(axis=1))
