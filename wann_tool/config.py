from collections import namedtuple

Game = namedtuple('Game', ['env_name', 'input_size', 'output_size', 'wann_file', 'action_select', 'weight_bias'])

games = {}

# evaluate on the Kaggle spam test set
spamtest = Game(env_name='SpamTEST',
  input_size=128,
  output_size=2,
  wann_file='spam_ascii_mean.out',
  action_select='softmax',
  weight_bias=0.0,
)
games['spamtest'] = spamtest

# evaluate on the iMDb review test set
imdbtest = Game(env_name='iMDbTEST',
  input_size=128,
  output_size=2,
  wann_file='',
  action_select='softmax',
  weight_bias=0.0
)
games['imdbtest'] = imdbtest