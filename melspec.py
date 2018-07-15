import sys
import os.path
from PIL import Image
import numpy
#import scipy.fftpack
#import scipy.signal
#import scipy.io.wavfile
#import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
"""
def db_array_to_grayscale_bytes(db_array, lower_db, upper_db):
    db_array    = numpy.clip(db_array, lower_db, upper_db)
    db_depth    = upper_db - lower_db
    grayscales  = numpy.multiply(numpy.divide(numpy.subtract(db_array, lower_db), db_depth), 255)
    return bytes(grayscales.astype(numpy.int8))
"""
#key_input=input("fs=11025に限る．  ./specwav/  :  filename>>>")
key_input="Y_dkeW6lqmq4_30.000_40.000.wav"
data, sr = librosa.load("./specwav/" + key_input,mono=True)
nn=np.floor(len(data)/sr)
nfft=int(sr*0.08)
overlap=int(nfft/2)
t=126*(overlap/sr)+(nfft/sr)
offset=int(t*sr)
ttt=np.floor(nn/t)
if ttt==1:
    ttt=ttt+1
for i in range(int(ttt-1)):
    y=data[i*offset:(i+1)*offset-1]
    S = np.array(librosa.feature.melspectrogram(y, sr=sr,n_fft=nfft,hop_length=overlap))
#    S[S<1]=1
    logS = librosa.power_to_db(S, ref=np.max)
#Normalize to [0,255]
    S_norm=logS/np.amax(logS)
    Norma_log_S=np.around((S_norm*255))
#        grayscale_bytes_array = []↲
#        grayscale_bytes_array.append(db_array_to_grayscale_bytes(amp_db[:window_half_size], -120, 0))
#    plt.figure()
#    plt.subplot(log_S)
#    y=librosa.display.specshow(log_S)
#    plt.show()
    basename = os.path.basename(key_input)
    spectrogram_file_name ="./spectrogram/" + "___"+ str(i) +"___" + basename + "_spectrogram.png"
#        with open(spectrogram_file_name, "wb") as pgm_file:
#    log_S = Image.fromarray(log_S)
#    if log_S.mode != 'RGB':
#        log_S = log_S.convert('RGB')
#    log_S.save(spectrogram_file_name)
    pil_img_gray = Image.fromarray(np.uint8(Norma_log_S))
    pil_img_gray.save(spectrogram_file_name)
