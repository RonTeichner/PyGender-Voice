#test_gender.py
import os
#import cPickle
import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

enableFigure = False

def get_MFCC(sr,audio):
    features = mfcc.mfcc(audio,sr, 0.025, 0.01, 13, appendEnergy = False)
    # 0.025 - the length of the analysis window is 25ms (400 samples)
    # 0.01 - the step between successive windows 10ms (160 samples)
    # 13 - the number of cepstrum coefficients
    feat     = np.asarray(())
    for i in range(features.shape[0]):
        temp = features[i,:]
        if np.isnan(np.min(temp)):
            continue
        else:
            if feat.size == 0:
                feat = temp
            else:
                feat = np.vstack((feat, temp))
    features = feat
    features = preprocessing.scale(features)
    return features

#path to testing data
#sourcepath = "./pygender/test_data/AudioSet/female_clips/"
sourcepath = "./pygender/test_data/AudioSet/male_clips/"
#path to saved models    
modelpath  = "./pygender/"

gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]
models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
genders   = [fname.split("/")[-1].split(".gmm")[0] for fname
              in gmm_files]
files     = [os.path.join(sourcepath,f) for f in os.listdir(sourcepath) 
              if f.endswith(".wav")] 

maxAudioLength = 0
for f in files:
    sr, audio = read(f)
    maxAudioLength = np.maximum(maxAudioLength, audio.shape[0]) # samples

maxAudioLength = maxAudioLength / sr # sec
likelihoodMatrix = np.zeros((int(np.ceil(maxAudioLength / 10e-3)), len(files), len(models)))
nFemales = 0
nMales = 0
for fIdx, f in enumerate(files):
    sr, audio  = read(f)
    features   = get_MFCC(sr,audio)
    scores     = None
    log_likelihood = np.zeros(len(models)) 
    for i in range(len(models)):
        gmm    = models[i]         #checking with each model one by one
        scores = np.array(gmm.score_samples(features))
        log_likelihood_alongTime = scores.cumsum()
        log_likelihood[i] = scores.sum()
        likelihoodMatrix[:scores.shape[0], fIdx, i] = log_likelihood_alongTime
        likelihoodMatrix[scores.shape[0]:, fIdx, i] = likelihoodMatrix[scores.shape[0]-1, fIdx, i]


    winner = np.argmax(log_likelihood)

    if genders[winner] == 'female':
        nFemales += 1
    else:
        nMales += 1

    if enableFigure:
        print(f.split("/")[-1])
        print("\tdetected as - ", genders[winner], "\n\tscores:female ", log_likelihood[0], ",male ", log_likelihood[1], "\n")
        if winner == 1:
            likelihoodDiff = likelihoodMatrix[:scores.shape[0], fIdx, 1] - likelihoodMatrix[:scores.shape[0], fIdx, 0]
        else:
            likelihoodDiff = likelihoodMatrix[:scores.shape[0], fIdx, 0] - likelihoodMatrix[:scores.shape[0], fIdx, 1]
        likelihoodDiff = 10 * likelihoodDiff / np.log(10)
        tVec = np.arange(0, scores.shape[0]) * 10e-3 + (25e-3) / 2
        #alpha = likelihoodMatrix[:scores.shape[0], fIdx, 0]
        #beta = likelihoodMatrix[:scores.shape[0], fIdx, 1]
        #gender0_prob = np.exp(alpha) / (np.exp(alpha) + np.exp(beta))
        #gender1_prob = np.exp(beta) / (np.exp(alpha) + np.exp(beta))

        #plt.subplot(2,1,1)
        plt.plot(tVec, np.minimum(likelihoodDiff,100))
        plt.xlabel('sec')
        plt.ylabel('db')
        plt.title('detected as - ' + genders[winner] +'; logLikelihood diff')

        #plt.subplot(2,1,2)
        #plt.plot(tVec, gender0_prob, label=genders[0])
        #plt.plot(tVec, gender1_prob, label=genders[1])
        #plt.legend()
        #plt.xlabel('sec')
        #plt.ylabel('probs')
print('nFiles = %d; nMale = %d, nFemale = %d' % (len(files), nMales, nFemales))
meanMaleLikelihood = (likelihoodMatrix[:, :, 1] - likelihoodMatrix[:, :, 0]).mean(axis=1)
meanMaleLikelihood = 10 * meanMaleLikelihood / np.log(10)
meanFemaleLikelihood = (likelihoodMatrix[:, :, 0] - likelihoodMatrix[:, :, 1]).mean(axis=1)
meanFemaleLikelihood = 10 * meanFemaleLikelihood / np.log(10)

tVec = np.arange(0, meanFemaleLikelihood.shape[0]) * 10e-3 + (25e-3) / 2
plt.figure()
plt.subplot(1,2,1)
plt.plot(tVec, meanFemaleLikelihood)
plt.xlabel('sec')
plt.title('mean female likelihood [db]')

plt.subplot(1,2,2)
plt.plot(tVec, meanMaleLikelihood)
plt.xlabel('sec')
plt.title('mean male likelihood [db]')

plt.show()
