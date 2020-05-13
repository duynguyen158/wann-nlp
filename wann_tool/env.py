import numpy as np
import gym

def make_env(env_name, encoder, max_features, seed=-1, render_mode=False):

  if (env_name.startswith("Spam")):
    print("KaggleSpam_started")
    from custom_envs.classify_gym import ClassifyEnv
    if env_name.startswith("SpamTEST"):
      from custom_envs.classify_gym import spam_test
      test_sentences, test_labels = spam_test(encoder, max_features)
      env = ClassifyEnv(test_sentences, test_labels, batch_size=1114, accuracy_mode=True)
    elif env_name.startswith("SpamTRAIN"):
      from custom_envs.classify_gym import spam_train
      train_sentences, train_labels = spam_train(encoder, max_features)
      env = ClassifyEnv(train_sentences, train_labels, batch_size=4458, accuracy_mode=True)

  elif (env_name.startswith("iMDb")):
    print("iMDbReview_started")
    from custom_envs.classify_gym import ClassifyEnv
    if env_name.startswith("iMDbTEST"):
      from custom_envs.classify_gym import imdb_test
      test_sentences, test_labels = imdb_test(encoder, max_features)
      env = ClassifyEnv(test_sentences, test_labels, batch_size=25000, accuracy_mode=True)
    elif env_name.startswith("iMDbTRAIN"):
      from custom_envs.classify_gym import imdb_train
      train_sentences, train_labels = imdb_train(encoder, max_features)
      env = ClassifyEnv(train_sentences, train_labels, batch_size=25000, accuracy_mode=True)


  if (seed >= 0):
    env.seed(seed)
  
  '''
  print("environment details")
  print("env.action_space", env.action_space)
  print("high, low", env.action_space.high, env.action_space.low)
  print("environment details")
  print("env.observation_space", env.observation_space)
  print("high, low", env.observation_space.high, env.observation_space.low)
  #assert False
  '''
  
  return env
