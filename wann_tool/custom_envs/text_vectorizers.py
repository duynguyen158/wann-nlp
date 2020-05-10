import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer

# --- ASCII ---
class ASCIIVectorizer():
  '''
  A "bad" embedding style that restricts the amount of information going in.

  Words (tokens) are split into characters, each of which is then converted into 
  its respective extended-ASCII codes (0 -> 255).

  A token's representation is then calculated either by taking the encoding of its 
  first character or the average encoding of all of its characters.
  '''
  def __init__(self, max_features, char_aggregator="first"):
    self.max_features = max_features
    self.char_aggregator = char_aggregator
    #print(f"...ASCII vectorizer -- Shuffled encodings: {use_shuffle}, Aggregation method: {char_aggregator}...")

  def transform(self, texts, labels):
    '''
    Splits tokens, convert characters, aggregates encodings, and pad/trim 
    sequences

    The output matrix has size `[data_length x num_features]`
    '''
    #print("...Splitting words & aggregating chars...")
    preprocessed = preprocess(texts)
    first_char_vectorizer = lambda data, func: [[func(w[0]) for w in t] for t in data]
    mean_vectorizer = lambda data, func: [[np.mean([func(c) for c in w]) for w in t] for t in data]

    func = lambda c: ord(c) if ord(c) <= 255 else 0

    if self.char_aggregator == "first":
      transformed = mean_vectorizer(preprocessed, func)
    else:
      transformed = first_char_vectorizer(preprocessed, func)

    z = np.array(collate_func(transformed, self.max_features)) / 255
    labels = np.array(labels)
    return z, labels

# --- BoW ---
class BoWVectorizer(CountVectorizer):
  '''
  Simple bag-of-word vectorizer that uses training data as a corpus

  To vary amount of encoded information, change vocab size, 
  i.e. `max_features`
  '''
  def __init__(self, max_features):
    super().__init__(max_features=max_features)

  def transform(self, texts, labels):
    z = super().transform(texts).toarray()
    row_max = np.amax(z, axis=1)
    row_max = np.where(row_max==0, 1, row_max)
    z = z / row_max[:, None]
    labels = np.array(labels)
    return z, labels

# --- BiLSTM ---
class BiLSTMVectorizer():
  '''
  Pre-trained BiLSTM sentence encoder from Conneau et al.
  - Paper: https://arxiv.org/pdf/1705.02364.pdf
  - Code: https://github.com/facebookresearch/InferSent

  To vary amount of encoded information, change vocab size

  Make sure to run `python -c "import nltk; nltk.download('punkt')"` 
  the first time
  '''
  def __init__(self, vocab_size, pooling="max"):
    from domain.InferSent.models import InferSent
    import torch
    self.params_model = {'bsize': 64, \
                         'word_emb_dim': 300, \
                         'enc_lstm_dim': 2048, \
                         'pool_type': pooling, \
                         'dpout_model': 0.0, \
                         'version': 1}
    self.model = InferSent(self.params_model)
    if torch.cuda.is_available():
      self.model = self.model.cuda() 

    MODEL_PATH = "domain/InferSent/encoder/infersent1.pkl"
    self.model.load_state_dict(torch.load(MODEL_PATH))
    W2V_PATH = "domain/InferSent/GloVe/glove.840B.300d.txt"
    self.model.set_w2v_path(W2V_PATH)
    self.model.build_vocab_k_words(vocab_size)

  def transform(self, texts, labels):
    z = self.model.encode(texts)
    labels = np.array(labels)
    return z, labels

# --- Utilities ---
def preprocess(text_data):
  '''
  Returns a list of lists of preprocessed tokens for each sequence
  '''
  #print("...Splitting sequences into words...")
  import spacy as sc 
  # Make sure to run `spacy download en_core_web_sm` in command line first
  enModel = sc.load("en_core_web_sm")
  preprocessed = [[token.text for token in enModel(d)] for d in text_data]
  return preprocessed

def collate_func(data, max_features=128):
  '''
  Dynamically pads or trims each sequence 
  so that all data have the same length.
  '''
  #print("...Padding/trimming sequences to equal lengths...")
  # Stores padded/trimmed sequences
  data_list = []
  # Loop through every pair (X, y) in batch
  for seq in data:
    if len(seq) < max_features:
      # Pad X with zeroes
      seq.extend([0 for i in range(max_features - len(seq))])
    else:
      # Trim X
      seq = seq[0:max_features]
    data_list.append(seq)
  return data_list
