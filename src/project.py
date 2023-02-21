###
### ISS project     (2021/2022)
### 
### Horky Dominik   (xhorky32)
### 3BIT, FIT VUT
###

### Requirements:
###
### 1) python3      
### 2) numpy
### 3) scipy
###

### This script is priting to stdout only things, that are required by assignment (without verbose enabled). 

try:
    import numpy as np
except ModuleNotFoundError:
    print("ERROR: You have not installed the 'NumPy' module for Python, which is required to run this script.\nTry: 'pip install numpy'")
    exit(1)
try:
    import matplotlib.pyplot as pt
except ModuleNotFoundError:
    print("ERROR: You have not installed the 'matplotlib' module for Python, which is required to run this script.\nTry: 'pip install matplotlib'")
    exit(1)
import wave
import math
try:
    import scipy.signal as ss
except ModuleNotFoundError:
    print("ERROR: You have not installed the 'scipy' module for Python.")
    exit(1)

###
#####   DECLARATIONS
###

audioPath = "../audio"
audioFreq = 16*1000
audioBitWidth = 16
recordPath = audioPath + "/" + "xhorky32.wav"

verbose = True                                      # more talking                                          (default: True)
plotGraphs = True                                   # plotting graphs to outputPath and saving as image(s)  (default: True)
graphDPI = 200                                      # DPI resolution of plotted graphs                      (default: 200)
outputPath = "../out"                               # path where graph images will be saved                 (default: "../out")

###
#####   FUNCTIONS
###

# prints text only when verbose is enabled (dev purposes)
def verbosePrint(x):
    if verbose:
        print(x)

# returns text-string value depending on bool variable value
def switchTextOnBool(bool, textTrue, textFalse):
    switch = textFalse
    if bool:
        switch = textTrue
    return switch

# initialize global variables
def init():
    verbosePrint("----------")
    verbosePrint("Verbose print:\t\t\tON")
    verbosePrint("Plotting graphs:\t\t"+switchTextOnBool(plotGraphs, "YES", "NO")+"\t(directory: '"+outputPath+"')")
    verbosePrint("----------\n\n")
    global signal
    global time
    global samples
    global seconds
    signal = 0
    time = 0
    samples = 0
    seconds = 0

# https://newbedev.com/python-write-a-wav-file-into-numpy-float-array
# load signal, basic info
def basics():
    global time
    global signal
    global samples
    global seconds
    global frames
    global audioChannels
    global audioSampWidth
    recordFile = wave.open(recordPath)
    samples = recordFile.getnframes()
    frames = recordFile.getframerate()  # 16 kHz
    audioChannels = recordFile.getnchannels()
    audioSampWidth = recordFile.getsampwidth()
    seconds = samples / frames
    print("Signal xhorky32.wav loaded.\n Samples: " + str(samples) + "\n Length: " + str(seconds) + " s")
    audio = recordFile.readframes(samples)
    signal16 = np.frombuffer(audio, dtype=np.int16)
    signal = signal16.astype(np.float32)
    time = np.arange(0,seconds,seconds/len(signal))
    print(" Maximum value: " + str(max(signal)) + "\n Minimum value: " + str(min(signal)) +"\n")
    recordFile.close()
    plotSimpleGraph("Vstupní signál", time, signal, "čas [sekundy]", "hodnoty signálu")

# normalization, framing
def normalize():
    global normalized_signal
    int16_max = 2**15   # 2^15
    normalized_signal = signal/int16_max
    verbosePrint("Input signal has been normalized.\n Maximum value: " + str(max(normalized_signal)) + "\n Minimum value " + str(min(normalized_signal)))
    framesCnt = int(samples / 512) + 1
    global framedSignal
    global fS
    framedSignal = []
    i = 0
    while (i < framesCnt):
        framedSignal.insert(i, [])
        newArr = framedSignal[i]
        for k in range((i*512),(i*512)+1024):
            if k < samples:
                newArr.insert(k, normalized_signal[k])
            else:
                newArr.insert(k, 0)     # if frame is
        i = i + 1 
    
    fS = 7    # selected frame (indexing from 0)
    frameSeconds = ( ( seconds / samples ) * 1024 ) * 1000
    frameTime = np.arange(fS*frameSeconds / 2, fS*frameSeconds/2+frameSeconds, frameSeconds/(len(framedSignal[fS])))
    print("Selected frame number "+str(fS)+ ".\n Time: "+str(fS*frameSeconds / 2)+" - "+str(fS*frameSeconds/2+frameSeconds)+" ms")
    plotSimpleGraph("Vybraný rámec", frameTime/1000, framedSignal[fS], "čas [sekundy]", "hodnoty signálu")
    global selectedFrame
    selectedFrame = framedSignal[fS]

# Discrete Fourier Transform - mine and FFT (python) implementation
def DFT(frameNumber):
    global framedSignal
    global DFTresult

    # DFT implementation by Python
    result = (np.fft.fft(framedSignal[frameNumber]))
    plotSimpleGraph("DFT rámce (Python NumPy FFT)", np.arange(0, 16000, 16000/1024), abs(result), "frekvence [Hz]", "modul DFT")
    verbosePrint("\nFFT completed.")

    # mine
    x = framedSignal[frameNumber]
    X_ = list()
    for k in range(0, 1024):
        X_.append(0)
        for n in range(0, 1024):      # for x in range(a,b) means interating till x is [a, b), so no need for N-1 ; N = 1024
            X_[k] += x[n] * (math.cos(2*(math.pi) * k * n / 1024))

    module = list(map(abs, X_))
    plotSimpleGraph("DFT rámce", np.arange(0, 8000, 8000/512), module[:512], "frekvence [Hz]", "modul DFT")
    verbosePrint("DFT completed.\n")

# spectogram
# mostly from here: (https://nbviewer.org/github/zmolikova/ISS_project_study_phase/blob/master/Zvuk_spektra_filtrace.ipynb)
def spect():
    global sgr
    global f
    global t
    f, t, sgr = ss.spectrogram(normalized_signal, audioFreq, nperseg=1024, noverlap=512)
    sgr_log = 10*np.log10(sgr+1e-20)
    figx = pt.figure(figsize=(9,6))
    pt.pcolormesh(t,f,sgr_log)
    pt.xlabel('Čas [s]')
    pt.ylabel('Frekvence [Hz]')
    cbar = pt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

    pt.tight_layout()
    #pt.title("Spectrogram")
    figx.savefig(outputPath+"/spectrogram.png", dpi=graphDPI)
    verbosePrint("Spectogram created and plotted.\n")

# finding anomalous signal
def errors():
    global anomalous_freqs
    peakCol = 103   # almost last column
    part = sgr[:, peakCol]
    peakIndex, heights = ss.find_peaks(part, height=1e-7)
    step = (audioFreq/2) / part.shape[0]
    anomalous_freqs = peakIndex * step
    fig = pt.figure()
    print("Found anomalies in frequencies below:")
    for el in anomalous_freqs:
        print(" "+str(el)+" Hz")
    print("\nThese frequencies should be multiplies of first one:")
    for i in range(1,5):
        print(" "+str(anomalous_freqs[0]*i)+" Hz\t\t(~ "+str(700*i)+" Hz)")
    if not plotGraphs:
        return
    pt.plot(anomalous_freqs, part[peakIndex], "xr")
    pt.plot(np.arange(0, 8000, step), part)
    pt.xlabel("Frekvence [Hz]")
    pt.ylabel("Hodnota")
    pt.title("Rušivé frekvence")
    fig.savefig(outputPath+"/rusivy.png", dpi=graphDPI)

# generating signal with four cosines (anomalous frequencies)
def cosines():
    newSamples = []
    for i in range(samples):
        newSamples.append(i/audioFreq)
    
    cosines = []
    for j in range(0,4):    # (0-3)
        cosines.append(np.cos(2 * np.pi * anomalous_freqs[j] * np.array(newSamples)))
    
    mixed = cosines[0] + cosines[1] + cosines[2] + cosines[3]
    
    newAudio = wave.open("../audio/cosines.wav", mode='wb')
    newAudio.setframerate(frames)
    newAudio.setnchannels(audioChannels)
    newAudio.setsampwidth(audioSampWidth)
    newAudio.writeframes(mixed.astype(np.int16))
    newAudio.close()
    verbosePrint("\nGenerated new signal from four cosines. (saved in: ./audio/cosines.wav)")
    
    nf, nt, nsgr = ss.spectrogram(mixed.astype(np.int16), audioFreq, nperseg=1024, noverlap=512)
    sgr_log = 10*np.log10(nsgr+1e-20)
    figx = pt.figure(figsize=(9,6))
    pt.pcolormesh(nt,nf,sgr_log)
    pt.xlabel('Čas [s]')
    pt.ylabel('Frekvence [Hz]')
    cbar = pt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

    pt.tight_layout()
    figx.savefig(outputPath+"/cos_spectrogram.png", dpi=graphDPI)
    verbosePrint("Spectogram for 4 cosines created and plotted.")

    
# plotting graph function
def plotSimpleGraph(title, xList, yList, xLabel, yLabel):
    if not plotGraphs:
        return
    fig = pt.figure()
    if len(xList) != len(yList):
        print("Unable to plot graph. Size of input data differs (x: "+str(len(xList))+"; y: "+str(len(yList))+").")
        exit(1)
    pt.plot(xList, yList)
    pt.xlabel(xLabel)
    pt.ylabel(yLabel)
    pt.title(title)
    pt.show()
    fig.savefig(outputPath+"/"+title+".png", dpi=graphDPI)

###
#####   MAIN
###

init()

basics()    # 4.1
normalize() # 4.2
DFT(fS)     # 4.3
spect()     # 4.4
errors()    # 4.5
cosines()   # 4.6

verbosePrint("\nScript ended succesfully.")