import numpy as np
import gym

def make_env(env_name, seed=-1, render_mode=False):
  if(env_name.startswith("Spam")):
    print("KaggleSpam_started")
    from custom_envs.classify_gym import ClassifyEnv, kaggle_spam_train, kaggle_spam_test
    if env_name.startswith("SpamTEST"):
      test_sentences, test_labels = kaggle_spam_test()
      env = ClassifyEnv(test_sentences, test_labels, batch_size=1114, accuracy_mode=True)
    elif env_name.startswith("SpamTRAIN"):
      train_sentences, train_labels = kaggle_spam_train()
      env = ClassifyEnv(train_sentences, train_labels, batch_size=4458, accuracy_mode=True)
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
