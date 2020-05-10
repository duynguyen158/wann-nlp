import numpy as np
import gym
from matplotlib.pyplot import imread


def make_env(env_name, seed=-1, render_mode=False):
    
  # -- Classification ------------------------------------------------ -- #
  if (env_name.startswith("Classify")):
    from domain.classify_gym import ClassifyEnv
    if env_name.endswith("spam"):
      from domain.classify_gym import kaggle_spam
      trainSet, target = kaggle_spam()

    env = ClassifyEnv(trainSet,target)  

  if (seed >= 0):
    domain.seed(seed)

  return env