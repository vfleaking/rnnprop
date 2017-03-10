# RNNprop
## Training
You can use
```
python main.py --task rnnprop
```
to reproduce our RNNprop model, or use
```
python main.py --task deepmind-lstm-avg
```
to reproduce the DMoptimizer for comparison.

## Evaluation
To evaluate the performance of a trained model, use
```
python main.py --train optimizer_train_optimizee
```
with other command-line flags:
* `task`: Must be specified, `rnnprop` or `deepmind-lstm-avg`.
* `id`: Must be specified, the unique 6 digit letter string that represents a trained model.
* `eid`: Must be specified, the epoch to restore the model.
* `n_steps`: Steps to train the optimizee.
* `n_epochs`: How many times to train the optimizee, `0` means do not stop until keyboard interrupted.

## Optimizees
The optimizees used in all experiments are listed in `test_list.py`.
You can train them with the best traditional optimization algorithm by using
```
python main.py --train optimizee
```
with other command-line flags:
* `task`: Must be specified, a name in `test_list.py`, e.g., `mnist-nn-sigmoid-100`.
* `n_epochs`: How many times to train the optimizee, `0` means do not stop until keyboard interrupted.
