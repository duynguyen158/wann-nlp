from collections import namedtuple

Game = namedtuple('Game', ['env_name', 'input_size', 'output_size', 'wann_file', 'action_select', 'weight_bias'])

games = {}

# evaluate on the Kaggle spam test set
spamtest = Game(env_name='SpamTEST',
  input_size=128, # can vary
  output_size=2,
  wann_file='',
  action_select='softmax',
  weight_bias=0.0,
)
games['spamtest'] = spamtest

# evaluate on the Kaggle spam training set
spamtrain = Game(env_name='SpamTRAIN',
  input_size=128, # can vary
  output_size=2,
  wann_file='',
  action_select='softmax',
  weight_bias=0.0,
)
games['spamtrain'] = spamtrain

# evaluate on the iMDb review test set
imdbtest = Game(env_name='iMDbTEST',
  input_size=128, # can vary
  output_size=2,
  wann_file='',
  action_select='softmax',
  weight_bias=0.0
)
games['imdbtest'] = imdbtest

# evaluate on the iMDb review training set
imdbtrain = Game(env_name='iMDbTRAIN',
  input_size=128, # can vary
  output_size=2,
  wann_file='',
  action_select='softmax',
  weight_bias=0.0
)
games['imdbtrain'] = imdbtrain