import argparse    # 1. argparse?? import????.
import os
import sys
import random
import numpy as np
import jax
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from BGF_AR.training.main import main
import ast
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
dict_ = {}

parser = argparse.ArgumentParser(description='use argparse to run main')    # 2. parser

# 3. parser.add_argument
parser.add_argument('--cuda', default=0,  type=int)

parser.add_argument('--seed', default=0,  type=int)
parser.add_argument('--model_init_seed', default=0,  type=int)

parser.add_argument('--batch_size', default=128,  type=int)
parser.add_argument('--sequence_length', default=40,  type=int)
parser.add_argument('--hidden_size', default=256,  type=int)
parser.add_argument('--memory_cell_size', default=8,  type=int)
parser.add_argument('--memory_size', default=40,  type=int)
parser.add_argument('--is_autoregressive', default=False,  type=bool)

parser.add_argument('--folder_name', default='debug',  type=str)
parser.add_argument('--training_steps', default=1_000_000,  type=int)

parser.add_argument('--task', required=True,  type=str)
parser.add_argument('--architecture', required=True,  type=str)

parser.add_argument('--lr', default=5e-4,  type=float)

parser.add_argument('--max_range_test_length', default=100,  type=int)
parser.add_argument('--range_test_sub_batch_size', default=128,  type=int)

parser.add_argument('--valid_seed', default=0,  type=int)
parser.add_argument('--valid_length', default=100,  type=int) #default 100

parser.add_argument('--optim', default='none',  type=str)
parser.add_argument('--weight_a', default=1.0,  type=float)
parser.add_argument('--weight_b', default=1.0,  type=float)
parser.add_argument('--ema_sm', default=0.98,  type=float)
parser.add_argument('--queue_size', default=100,  type=int)  # queue size (lambda) of the queue-based BGF (optim ours_balance / ours_add)

parser.add_argument('--save_dir', default='none',  type=str)
parser.add_argument('--save_param', default=0,  type=int)  # 0: no checkpoints / 1: save params at steps [0,10,100,...,final] + final optimizer state / 2: dense schedule
parser.add_argument('--save_grad_signal', default=0,  type=int)  # 1: record 500 sampled gradient coordinates at every step -> grad_signal.pkl (see run_grad_visualization/)

parser.add_argument('--momentum', default=0.9,  type=float)
parser.add_argument('--base_optim', default='adam',  type=str, choices=['adam', 'adamw'])  # underlying optimizer of the baseline and every BGF variant
parser.add_argument('--weight_decay', default=1e-4,  type=float)  # decoupled weight decay (used when --base_optim adamw)

args = parser.parse_args()    # 4.

# here change args
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.cuda) #@
dict_['batch_size'] = args.batch_size
dict_['sequence_length'] = args.sequence_length
dict_['task'] = args.task
dict_['architecture'] = args.architecture
dict_['hidden_size'] = args.hidden_size
dict_['memory_cell_size'] = args.memory_cell_size
dict_['folder_name'] = args.folder_name

dict_['is_autoregressive'] = args.is_autoregressive
dict_['seed'] = args.seed
dict_['model_init_seed'] = args.model_init_seed
dict_['training_steps'] = args.training_steps
dict_['lr'] = args.lr
dict_['max_range_test_length'] = args.max_range_test_length
dict_['valid_seed'] = args.valid_seed
dict_['valid_length'] = args.valid_length

dict_['optim'] = args.optim
dict_['weight_a'] = args.weight_a
dict_['weight_b'] = args.weight_b
dict_['ema_sm'] = args.ema_sm
dict_['filter_step'] = args.queue_size

dict_['save_dir'] = args.save_dir
dict_['save_param'] = args.save_param
dict_['save_grad_signal'] = args.save_grad_signal

dict_['momentum'] = args.momentum
dict_['base_optim'] = args.base_optim
dict_['weight_decay'] = args.weight_decay

arch = args.architecture
arch_candidate = ['rnn', 'lstm', 'stack_rnn', 'ndstack_rnn', 'stack_lstm', 'tape_rnn', 'transformer_encoder_sincos',
                  'transformer_sincos', 'transformer_encoder_none', 'transformer_none', 'transformer_encoder_alibi',
                  'transformer_alibi', 'transformer_encoder_relative', 'transformer_relative', 'transformer_encoder_rotary',
                  'transformer_rotary',]
assert arch in arch_candidate

task = args.task
task_candidate = ['modular_arithmetic', 'parity_check', 'even_pairs', 'cycle_navigation',
                  'modular_arithmetic_brackets', 'reverse_string', 'missing_duplicate_string', 'duplicate_string',
                  'binary_addition', 'binary_multiplication', 'compute_sqrt', 'odds_first', 'solve_equation', 'stack_manipulation',
                  'bucket_sort',

                  'up_abab', 'up_add', 'up_alternating', 'up_dihedral', 'up_flipflop', 'up_gridworld', 'up_quaternion',
                  'up_symmetric', 'up_permutation_reset',
                  'up_dykefull_simple', 'up_dyketop_simple', 'up_dykerecog_simple', 'up_dykefull_complex',
                  'up_dyketop_complex', 'up_dykerecog_complex'
                  ]
assert task in task_candidate


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

set_global_seed(args.seed)
main(dict_)
