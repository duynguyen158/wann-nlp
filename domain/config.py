from collections import namedtuple
import numpy as np

Game = namedtuple('Game', ['env_name', 'time_factor', 'actionSelect',
  'input_size', 'output_size', 'layers', 'i_act', 'h_act',
  'o_act', 'weightCap','noise_bias','output_noise','max_episode_length','in_out_labels'])

games = {}

# -- Spam Classification ------------------------------------------------ -- #

spam = Game(env_name='Classify_spam',
  actionSelect='softmax', # all, soft, hard
  input_size=8,
  output_size=2,
  time_factor=0,
  layers=[128,9],
  i_act=np.full(8,1),
  h_act=[1,3,4,5,6,7,8,9,10], # No step function
  o_act=np.full(2,1),
  weightCap = 2.0,
  noise_bias=0.0,
  output_noise=[False, False, False],
  max_episode_length = 0,
  in_out_labels = []
)
L = [list(range(1, spam.input_size)),\
     list(range(0, spam.output_size))]
label = [item for sublist in L for item in sublist]
spam = spam._replace(in_out_labels=label)
games['spam'] = spam


# -- Binary Sentiment Analysis ------------------------------------------------ -- #

imdb = spam._replace(\
  env_name='Classify_imdb', input_size=128, output_size=2,\
  i_act=np.full(128,1), o_act=np.full(2,1))
L = [list(range(1, imdb.input_size)),\
     list(range(0, imdb.output_size))]
label = [item for sublist in L for item in sublist]
imdb = imdb._replace(in_out_labels=label)
games['imdb'] = imdb
