import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import sys
import cv2
import math
from sklearn.metrics import accuracy_score, f1_score

class ClassifyEnv(gym.Env):
  """Classification as an unsupervised OpenAI Gym RL problem.
  Includes scikit-learn digits dataset, MNIST dataset
  """

  def __init__(self, trainSet, target, batch):
    """
    Data set is a tuple of 
    [0] input data: [nSamples x nInputs]
    [1] labels:     [nSamples x 1]

    Example data sets are given at the end of this file
    """

    self.t = 0          # Current batch number
    self.t_limit = 0    # Number of batches if you need them
    
    self.batch   = batch  # Number of examples per batch / Spam: 75 / iMDb: 400
    
    self.seed()
    self.viewer = None

    self.trainSet = trainSet
    self.target   = target

    nInputs = np.shape(trainSet)[1]
    high = np.array([1.0]*nInputs)
    self.action_space = spaces.Box(np.array(0,dtype=np.float32), \
                                   np.array(1,dtype=np.float32))
    self.observation_space = spaces.Box(np.array(0,dtype=np.float32), \
                                   np.array(1,dtype=np.float32))

    self.state = None
    self.trainOrder = None
    self.currIndx = None

  def seed(self, seed=None):
    ''' Randomly select from training set'''
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def reset(self):
    ''' Initialize State'''    
    #print('Lucky number', np.random.randint(10)) # same randomness?
    self.trainOrder = np.random.permutation(len(self.target))
    self.t = 0 # timestep
    self.currIndx = self.trainOrder[self.t:self.t+self.batch]
    self.state = self.trainSet[self.currIndx,:]
    return self.state
  
  def step(self, action):
    ''' 
    Judge Classification, increment to next batch
    action - [batch x output] - softmax output
    '''
    y = self.target[self.currIndx]
    m = y.shape[0]

    #p = np.argmax(action, axis=1)
    #print(accuracy_score(y, p))

    accuracy = np.mean(np.argmax(action, axis=1) == y)
    log_likelihood = -np.log(action[range(m),y])
    loss = np.sum(log_likelihood) / m
    reward = -loss

    if self.t_limit > 0: # We are doing batches
      reward *= (1/self.t_limit) # average
      self.t += 1
      done = False
      if self.t >= self.t_limit:
        done = True
      self.currIndx = self.trainOrder[(self.t*self.batch):\
                                      (self.t*self.batch + self.batch)]

      self.state = self.trainSet[self.currIndx,:]
    else:
      done = True

    obs = self.state
    return obs, reward, done, {}, accuracy


# -- Data Sets ----------------------------------------------------------- -- #

# SPAM CLASSIFICATION
def spam(embedding_style, max_features):
  '''
  Return Kaggle spam training data 
  (embeddings & labels)
  '''
  import pandas as pd
  print("...Reading data...")
  train = pd.read_csv("data/spam_classify/train.csv")
  train_texts, train_labels = list(train.v2), list(train.v1)
  if embedding_style == "ascii":
    # ASCII
    print("spam_ascii_" + str(max_features))
    from domain.text_vectorizers import ASCIIVectorizer
    vectorizer = ASCIIVectorizer(max_features)
    z, labels = vectorizer.transform(train_texts, train_labels)
  elif embedding_style == "count":
    # BoW
    print("spam_count_" + str(max_features))
    from domain.text_vectorizers import BoWVectorizer
    vectorizer = BoWVectorizer(max_features=max_features)
    vectorizer.fit(train_texts)
    z, labels = vectorizer.transform(train_texts, train_labels) 
  elif embedding_style == "lstm":
    # BiLSTM mean/max pooling (default: max)
    print("spam_lstm_" + str(max_features))
    from domain.text_vectorizers import BiLSTMVectorizer
    vectorizer = BiLSTMVectorizer(vocab_size=max_features)
    z, labels = vectorizer.transform(train_texts, train_labels, bsize=32)
  return z, labels

# BINARY SENTIMENT ANALYSIS
def imdb(embedding_style, max_features):
  '''
  Return movie review training data 
  (embeddings & labels)
  '''
  import pandas as pd
  print("...Reading data...")
  train = pd.read_csv("data/sen_imdb/train.csv")
  train_texts, train_labels = list(train.text), list(train.pos)
  if embedding_style == "ascii":
    # ASCII
    print("imdb_ascii_" + str(max_features))
    from domain.text_vectorizers import ASCIIVectorizer
    vectorizer = ASCIIVectorizer(max_features)
    z, labels = vectorizer.transform(train_texts, train_labels)
    print(z.shape)
    print(z[0])
    print(labels.shape)
  elif embedding_style == "count":
    # BoW
    print("imdb_count_" + str(max_features))
    from domain.text_vectorizers import BoWVectorizer
    vectorizer = BoWVectorizer(max_features=max_features)
    vectorizer.fit(train_texts)
    z, labels = vectorizer.transform(train_texts, train_labels)
  elif embedding_style == "lstm":
    # BiLSTM mean/max pooling (default: max)
    print("imdb_lstm_" + str(max_features))
    from domain.text_vectorizers import BiLSTMVectorizer
    vectorizer = BiLSTMVectorizer(vocab_size=max_features)
    z, labels = vectorizer.transform(train_texts, train_labels, bsize=32)
  return z, labels


### Spectrogram dataset
import skimage.measure
from scipy.io import wavfile
from scipy import signal
from sklearn.model_selection import train_test_split
import glob

def create_spectrogram(file_name, window_size=20, step_size=10, eps=1e-10):
    """Creates a spectrogram from audio file"""
    sample_rate, audio = wavfile.read(file_name)
    
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    _, _, spec = signal.spectrogram(audio,  fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)

    # Create log spectrogram
    spectrogram = np.log(spec.astype(np.float32) + eps)

    # Max pooling
    spectrogram = skimage.measure.block_reduce(spectrogram, (13, 13), np.max)
    
    # Resize to 8x8 and flatten
    spectrogram = cv2.resize(spectrogram, (8,8), cv2.INTER_CUBIC).flatten()

    return spectrogram

 
def speech_mnist():
  print("Creating speech_mnist dataset")
  X = np.empty((2350*10, 64))
  y = np.empty((2350*10))
  numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
  for n, number in enumerate(numbers):
      paths = glob.glob(f"datasets/speech_mnist/{number}/*.wav")
      paths = sorted(paths)
      for i, path in enumerate(paths):
          X[n*2350+i,:] = create_spectrogram(path).flatten()
          y[n*2350+i] = n
  Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=123)
  print("done")
  return Xte, yte.astype(np.uint8)
  return Xtr, ytr.astype(np.uint8)

def speech_yesno():
  print("Creating speech_yesno dataset")
  X = np.empty((2375*2, 64))
  y = np.empty((2375*2))
  categories = ['no', 'yes']
  for i, cat in enumerate(categories):
      paths = glob.glob(f"datasets/speech_yesno/{cat}/*.wav")
      paths = sorted(paths)
      print(len(paths))
      for j, path in enumerate(paths):
          X[i*2375+j,:] = create_spectrogram(path).flatten()
          y[i*2375+j] = i
  Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=123)
  print("done")
  return Xte, yte.astype(np.uint8)
  return Xtr, ytr.astype(np.uint8)
  