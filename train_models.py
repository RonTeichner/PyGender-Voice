#train_models.py

import os
#import cPickle
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
#from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

n_components = 3
n_MFCC = 7

def get_MFCC(sr,audio,n_MFCC):
    features = mfcc.mfcc(audio,sr, 0.025, 0.01, n_MFCC, appendEnergy = False)
    features = preprocessing.scale(features)
    return features

#path to training data
source   = "./pygender/train_data/youtube/male/"
#source   = "./pygender/train_data/youtube/female/"
#path to save trained model
dest     = "./pygender/"
files    = [os.path.join(source,f) for f in os.listdir(source) if 
             f.endswith('.wav')] 
features = np.asarray(());

for f in files:
    sr,audio = read(f)
    vector   = get_MFCC(sr, audio, n_MFCC)
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))


gmm = GaussianMixture(n_components=n_components, max_iter = 200, covariance_type='diag', n_init = 3)
gmm.fit(features)
picklefile = f.split("/")[-2].split(".wav")[0]+".gmm"

# model saved as male.gmm
cPickle.dump(gmm,open(dest + picklefile,'wb'))
print('modeling completed for gender:',picklefile)
