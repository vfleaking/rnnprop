# RNNprop

Compatible with TensorFlow 0.12

## Training
You can use
```
python main.py --task rnnprop
```
to reproduce our RNNprop model, or use
```
python main.py --task deepmind-lstm-avg
```
to reproduce the DMoptimizer [Andrychowicz et al., 2016](https://arxiv.org/abs/1606.04474) for comparison.

A random 6 digit letter string will be automatically generated as a unique id for each training process, and a folder named `<task-name>-<id>_data` will be created to place data and logs.

## Evaluation
To evaluate the performance of a trained model, use
```
python main.py --train optimizer_train_optimizee
```
with other command-line flags:
* `task`: Must be specified, `rnnprop` or `deepmind-lstm-avg`.
* `id`: Must be specified, the unique 6 digit letter string that represents a trained model.
* `eid`: Must be specified, the epoch to restore the model.
* `n_steps`: Steps to train the optimizee. (Default is 100)
* `n_epochs`: How many times to train the optimizee, `0` means do not stop until keyboard interrupted. (Default is 0)

## Optimizees
The optimizees used in all experiments are listed in `test_list.py`.
You can train them with the best traditional optimization algorithm by using
```
python main.py --train optimizee
```
with other command-line flags:
* `task`: Must be specified, a name in `test_list.py`, e.g., `mnist-nn-sigmoid-100`.
* `n_epochs`: How many times to train the optimizee, `0` means do not stop until keyboard interrupted. (Default is 0)

## License
MIT License.
