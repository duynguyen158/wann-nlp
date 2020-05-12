import numpy as np
import gym
#from matplotlib.pyplot import imread


def make_env(env_name, encoder, max_features, seed=-1, render_mode=False):
    
  # -- Classification ------------------------------------------------ -- #
  if (env_name.startswith("Classify")):
    from domain.classify_gym import ClassifyEnv
    if env_name.endswith("spam"):
      from domain.classify_gym import spam
      trainSet, target = spam(encoder, max_features)
    elif env_name.endswith("imdb"):
      from domain.classify_gym import imdb
      trainSet, target = imdb(encoder, max_features)

    env = ClassifyEnv(trainSet,target)  

  if (seed >= 0):
    domain.seed(seed)

  return env