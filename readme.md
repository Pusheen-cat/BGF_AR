# Balancing Gradient Frequencies Facilitates Inductive Inference in Algorithmic Reasoning

> **Acknowledgment:** This repository utilizes code from Google DeepMind's research on the Neural Networks Chomsky Hierarchy.
> * **Paper:** [Neural Networks and the Chomsky Hierarchy](https://arxiv.org/abs/2207.02098)
> * **Original Repository:** [google-deepmind/neural_networks_chomsky_hierarchy](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy)

This repository contains code to run and manage experiments across various model architectures and tasks.

---

## 🚀 Running a Single Experiment (`one_sample.py`)

`one_sample.py` is a helper script designed to run a single instance of `basic_parser.py` without complex loops, allowing you to easily specify the task and architecture you want to test.

### 1. Basic Execution
Run the experiment with default values (e.g., `modular_arithmetic`, `rnn`, `seed=3`).

```bash
python one_sample.py
```

### 2. Custom Execution
You can manually specify the task, model architecture, seed, learning rate, and other parameters.

```bash
python one_sample.py \
    --task parity_check \
    --architecture lstm \
    --seed 0 \
    --model_init_seed 0 \
    --valid_seed 0 \
    --lr 1e-3 \
    --training_steps 1000000 \
    --cuda 1
```

---

## ⚙️ `basic_parser.py` Arguments

Below is the description of the `argparse` arguments used in the main `basic_parser.py` script.

| Argument | Type |   Default   | Description                                                                                                                                                    |
| :--- | :---: |:-----------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **General & Environment** | |             |                                                                                                                                                                |
| `--cuda` | `int` |     `0`     | CUDA device ID to use (e.g., 0, 1, 2)                                                                                                                          |
| `--save_dir` | `str` | `Required` | Root directory path to save experiment results and model checkpoints                                                                                           |
| `--folder_name` | `str` |  `'debug'`  | Sub-folder name to store results for individual experiments                                                                                                    |
| **Seed Settings** | |             |                                                                                                                                                                |
| `--seed` | `int` |     `0`     | Global seed for data generation and random number control                                                                                                      |
| `--model_init_seed` | `int` |     `0`     | Seed for initializing model parameters                                                                                                                         |
| `--valid_seed` | `int` |     `0`     | Random seed for validation data generation                                                                                                                     |
| **Experiment & Task** | |             |                                                                                                                                                                |
| `--task` | `str` | `Required`  | Name of the task to run (e.g., `modular_arithmetic`, `parity_check`) - see **tasks**                                                                           |
| `--architecture` | `str` | `Required`  | Model architecture to use (e.g., `RNN`, `LSTM`, `Transformer_none`) - see **models**                                                                           |
| **Model Hyperparameters** | |             |                                                                                                                                                                |
| `--hidden_size` | `int` |    `256`    | Hidden dimension size of the model                                                                                                                             |
| `--memory_cell_size` | `int` |     `8`     | Number of memory stacks or tapes (`Stack-RNN` and `Tape-RNN`)                                                                                                  |
| `--memory_size` | `int` |    `40`     | Number of memory cells in each stack or tape (`Stack-RNN` and `Tape-RNN`)                                                                                      |
| `--is_autoregressive` | `bool` |   `False`   | Whether to use an autoregressive modeling approach                                                                                                             |
| **Training Hyperparameters**| |             |                                                                                                                                                                |
| `--training_steps` | `int` | `1,000,000` | Total number of training steps                                                                                                                                 |
| `--batch_size` | `int` |    `128`    | Batch size per training step                                                                                                                                   |
| `--sequence_length` | `int` |    `40`     | Length of the input sequence                                                                                                                                   |
| `--lr` | `float` |   `5e-4`    | Learning rate                                                                                                                                                  |
| `--optim` | `str` |  `'none'`   | Type of optimizer to use (`none`: default Adam, `ours_balance`: BGF with sliding window, `ours_ema`: BGF with EMA, `ours_add`: Non-balanced gradient addition) |
| `--momentum` | `float` |    `0.9`    | Momentum (beta1) value for the Adam optimizer                                                                                                                  |
| `--ema_sm` | `float` |   `0.98`    | Exponential Moving Average (EMA) smoothing parameter of `ours_ema`                                                                                             |
| `--weight_a` | `float` |    `1.0`    | Original gradient (high-frequency) proportion                                                                                                                  |
| `--weight_b` | `float` |    `1.0`    | Low-frequency gradient proportion                                                                                                                              |
| **Validation & Testing** | |             |                                                                                                                                                                |
| `--valid_length` | `int` |    `100`    | Sequence length for validation data                                                                                                                            |
| `--max_range_test_length` | `int` |    `100`    | Maximum length for length generalization tests                                                                                                                 |
| `--range_test_sub_batch_size`| `int` |    `128`    | Sub-batch size used during range testing                                                                                                                       |