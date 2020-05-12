
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
from custom_envs.text_vectorizers import ASCIIVectorizer
import numpy as np
import sys
import cv2
import math

class ClassifyEnv(gym.Env):

  def __init__(self, trainSet, target, batch_size=1000, accuracy_mode=False, f1_mode=False, recall_mode=False):
    """
    Data set is a tuple of 
    [0] input data: [nSamples x nInputs]
    [1] labels:     [nSamples x 1]

    Example data sets are given at the end of this file
    """

    self.t = 0          # Current batch number
    self.t_limit = 0    # Number of batches if you need them
    self.batch   = batch_size # Number of images per batch
    self.accuracy_mode = accuracy_mode
    self.f1_mode = f1_mode
    self.recall_mode = recall_mode

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

    if self.accuracy_mode:
      p = np.argmax(action, axis=1)
      accuracy = (float(np.sum(p==y)) / self.batch)
      reward = accuracy
    elif self.recall_mode: # Binary classification
      p = np.argmax(action, axis=1)
      tp = float(np.sum(np.logical_and(p==1,y==1)))
      fn = float(np.sum(np.logical_and(p==0,y==1)))
      recall = .0
      if tp + fn != 0:
        recall = tp / (tp + fn)
      reward = recall
    elif self.f1_mode: # Binary classification (e.g. spam)
      p = np.argmax(action, axis=1)
      tp = float(np.sum(np.logical_and(p==1,y==1)))
      tn = float(np.sum(np.logical_and(p==0,y==0)))
      fp = float(np.sum(np.logical_and(p==1,y==0)))
      fn = float(np.sum(np.logical_and(p==0,y==1)))
      precision, recall = .0, .0
      if tp + fp != 0:
        precision = tp / (tp + fp)
      if tp + fn != 0:
        recall = tp / (tp + fn)
      if precision + recall == 0:
        f1 = 0
      else:
        f1 = 2 * precision * recall / (precision + recall)
      reward = f1
    else:
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
    return obs, reward, done, {}


# -- Data Sets ----------------------------------------------------------- -- #

# Spam classification
def spam_test(embedding_style, max_features):
  '''
  Return Kaggle spam test data 
  (embeddings & labels)
  '''
  import pandas as pd
  print("...Reading test data...")
  test = pd.read_csv("../data/spam_classify/test.csv")
  test_texts, test_labels = list(test.v2), list(test.v1)
  if embedding_style == "ascii":
    # ASCII
    print("spam_ascii_" + str(max_features) + "_test")
    from custom_envs.text_vectorizers import ASCIIVectorizer
    vectorizer = ASCIIVectorizer(max_features)
    z, labels = vectorizer.transform(test_texts, test_labels)
  elif embedding_style == "count":
    # BoW
    print("spam_count_" + str(max_features) + "_test")
    from custom_envs.text_vectorizers import BoWVectorizer
    train = pd.read_csv("../data/spam_classify/train.csv")
    train_texts, train_labels = list(train.v2), list(train.v1)
    vectorizer = BoWVectorizer(max_features=max_features)
    vectorizer.fit(train_texts)
    z, labels = vectorizer.transform(test_texts, test_labels)
  elif embedding_style == "lstm":
    # BiLSTM mean/max pooling (default: max)
    print("spam_lstm_" + str(max_features) + "_test")
    from custom_envs.text_vectorizers import BiLSTMVectorizer
    vectorizer = BiLSTMVectorizer(vocab_size=max_features)
    z, labels = vectorizer.transform(test_texts, test_labels, bsize=32)
  return z, labels

def spam_train(embedding_style, max_features):
  '''
  Return Kaggle spam test data 
  (embeddings & labels)
  '''
  import pandas as pd
  print("...Reading training data...")
  train = pd.read_csv("../data/spam_classify/train.csv")
  train_texts, train_labels = list(train.v2), list(train.v1)
  if embedding_style == "ascii":
    # ASCII
    print("spam_ascii_" + str(max_features) + "_train")
    from custom_envs.text_vectorizers import ASCIIVectorizer
    vectorizer = ASCIIVectorizer(max_features)
    z, labels = vectorizer.transform(train_texts, train_labels)
  elif embedding_style == "count":
    # BoW
    print("spam_count_" + str(max_features) + "_train")
    from custom_envs.text_vectorizers import BoWVectorizer
    vectorizer = BoWVectorizer(max_features=max_features)
    vectorizer.fit(train_texts)
    z, labels = vectorizer.transform(train_texts, train_labels)
  elif embedding_style == "lstm":
    # BiLSTM mean/max pooling (default: max)
    print("spam_lstm_" + str(max_features) + "_train")
    from custom_envs.text_vectorizers import BiLSTMVectorizer
    vectorizer = BiLSTMVectorizer(vocab_size=max_features)
    z, labels = vectorizer.transform(train_texts, train_labels, bsize=32)
  return z, labels

# Binary Sentiment Analysis
def imdb_test(embedding_style, max_features):
  '''
  Return iMDb test data 
  (embeddings & labels)
  '''
  import pandas as pd
  print("...Reading data...")
  test = pd.read_csv("../data/sen_imdb/test.csv")
  test_texts, test_labels = list(test.text), list(test.pos)
  if embedding_style == "ascii":
    # ASCII
    print("imdb_ascii_" + str(max_features) + "_test")
    from custom_envs.text_vectorizers import ASCIIVectorizer
    vectorizer = ASCIIVectorizer(max_features)
    z, labels = vectorizer.transform(test_texts, test_labels)
  elif embedding_style == "count":
    # BoW
    print("imdb_count_" + str(max_features) + "_test")
    from custom_envs.text_vectorizers import BoWVectorizer
    train = pd.read_csv("../data/sen_imdb/train.csv")
    train_texts, train_labels = list(train.text), list(train.pos)
    vectorizer = BoWVectorizer(max_features=max_features)
    vectorizer.fit(train_texts)
    z, labels = vectorizer.transform(test_texts, test_labels)
  elif embedding_style == "lstm":
    # BiLSTM mean/max pooling (default: max)
    print("imdb_lstm_" + str(max_features) + "_test")
    from custom_envs.text_vectorizers import BiLSTMVectorizer
    vectorizer = BiLSTMVectorizer(vocab_size=max_features)
    z, labels = vectorizer.transform(test_texts, test_labels, bsize=32)
  return z, labels

def imdb_train(embedding_style, max_features):
  '''
  Return iMDb training data 
  (embeddings & labels)
  '''
  import pandas as pd
  print("...Reading training data...")
  train = pd.read_csv("../data/sen_imdb/train.csv")
  train_texts, train_labels = list(train.text), list(train.pos)
  if embedding_style == "ascii":
    # ASCII
    print("imdb_ascii_" + str(max_features) + "_train")
    from custom_envs.text_vectorizers import ASCIIVectorizer
    vectorizer = ASCIIVectorizer(max_features)
    z, labels = vectorizer.transform(train_texts, train_labels)
  elif embedding_style == "count":
    # BoW
    print("imdb_count_" + str(max_features) + "_train")
    from custom_envs.text_vectorizers import BoWVectorizer
    vectorizer = BoWVectorizer(max_features=max_features)
    vectorizer.fit(train_texts)
    z, labels = vectorizer.transform(train_texts, train_labels)
  elif embedding_style == "lstm":
    # BiLSTM mean/max pooling (default: max)
    print("imbd_lstm_" + str(max_features) + "_train")
    from custom_envs.text_vectorizers import BiLSTMVectorizer
    vectorizer = BiLSTMVectorizer(vocab_size=max_features)
    z, labels = vectorizer.transform(train_texts, train_labels, bsize=32)
  return z, labels