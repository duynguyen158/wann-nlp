# WANN tool
An adaption of [WANNTool](https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease/WANNTool) and [estool](https://github.com/hardmaru/estool)


## Setting things up

`pip install cma`


## Evaluating WANNs (single-weight)
For example, enter
```
python model.py spamtest -m ../log_spam/spam_bow_8_best.out -p spam_bow_8 --sweep 20 --lo -2 --hi 2
```
to evaluate the champion WANN of BoW embedding size 8 on the test spam set 20 times, each time with a single weight shared across all nodes. The weights are within range [-2, 2] and equidistant to one another, i.e. [-2, 1.8, -1.6, ..., 1.6, 1.8, 2]. Change `--sweep`, `--lo` and `--hi` to create different sets of weights.

More command-line arguments are available in `model.py`.


## Fine-Tune WANNs
Refer to [WANNTool](https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease/WANNTool). `-m` and `-p` arguments are also available to specify model path and model's original hyperparameters. We used CMA-ES as the optimization method with standard deviation 0.5, like Gaier and Ha.

Fine-tuned weights are stored in `log` as JSON files.


## Evaluating WANNs (fine-tuned)
Similar to single-weight configuration, but remove `--sweep` and `-f` to specify the path of JSON weight file.