"""Command-line entry point for a single input-length confounding toy run.

Only the task/architecture combinations that were actually used in the paper
are accepted: the 7 single-output tasks (the confounder requires
task.output_length == 1) and the 4 recurrent architectures.
"""
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from BGF_AR.run_toy_experiment.spu_len_main import main
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
dict_ = {}

parser = argparse.ArgumentParser(description='run one input-length confounding toy experiment')

parser.add_argument('--cuda', default=0,  type=int)

parser.add_argument('--batch_size', default=128,  type=int)
parser.add_argument('--sequence_length', default=40,  type=int)
parser.add_argument('--task', required=True,  type=str)
parser.add_argument('--architecture', required=True,  type=str)
parser.add_argument('--hidden_size', default=256,  type=int)
parser.add_argument('--memory_cell_size', default=8,  type=int)
parser.add_argument('--memory_size', default=40,  type=int)
parser.add_argument('--folder_name', default='debug',  type=str)

parser.add_argument('--is_autoregressive', default=False,  type=bool)

parser.add_argument('--seed', default=0,  type=int)
parser.add_argument('--model_init_seed', default=0,  type=int)
parser.add_argument('--training_steps', default=100_000,  type=int)

parser.add_argument('--lr', default=5e-4,  type=float)

parser.add_argument('--max_range_test_length', default=100,  type=int)
parser.add_argument('--range_test_sub_batch_size', default=128,  type=int)

parser.add_argument('--valid_seed', default=0,  type=int)
parser.add_argument('--valid_length', default=100,  type=int) #default 100

parser.add_argument('--optim', default='none',  type=str)  # none / ours_balance / ours_add
parser.add_argument('--weight_a', default=0.5,  type=float)
parser.add_argument('--weight_b', default=0.5,  type=float)
parser.add_argument('--queue_size', default=100,  type=int)  # queue size (lambda) of the queue-based BGF / ADD

parser.add_argument('--save_dir', default='./results_toy_length',  type=str)

parser.add_argument('--spu_prob', default=0.0,  type=float)  # spurious correlation strength alpha

args = parser.parse_args()

# here change args
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.cuda)
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
dict_['filter_step'] = args.queue_size

dict_['save_dir'] = args.save_dir

dict_['spu_prob'] = args.spu_prob

# Task/architecture combinations actually used in the paper's input-length
# confounding experiments (single-output tasks only).
arch_candidate = ['rnn', 'lstm', 'stack_rnn', 'tape_rnn']
assert args.architecture in arch_candidate, f'unsupported architecture {args.architecture!r}'

task_candidate = ['modular_arithmetic', 'parity_check', 'even_pairs', 'cycle_navigation',
                  'modular_arithmetic_brackets', 'missing_duplicate_string', 'solve_equation']
assert args.task in task_candidate, f'unsupported task {args.task!r}'


main(dict_)
