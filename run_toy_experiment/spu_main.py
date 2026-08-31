# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Train and evaluate a network on the label-concatenation toy experiment."""

import haiku as hk
import jax.numpy as jnp
import numpy as np

from BGF_AR.training import constants
from BGF_AR.training import curriculum as curriculum_lib
from BGF_AR.run_toy_experiment import spu_training as training
from BGF_AR.training import utils


import json
from datetime import datetime
import copy


##### This prevents pre-location of gpu memory
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#
######

def main(arg_dict={'developing':True}) -> None:
  # Change your hyperparameters here. See constants.py for possible tasks and
  # architectures.
  argv = copy.deepcopy(arg_dict)
  print('>>>> unused_argv', argv)
  keys = argv.keys()

  def check_dict(name, value):
      if name in keys:
          if value != argv[name]:
              print(f'>> Changed {name} : {value} to {argv[name]}')
          return argv[name]
      else:
          argv[name] = value
          return value

  spu_prob = check_dict('spu_prob', 0.0)
  batch_size = check_dict('batch_size',128)
  sequence_length = check_dict('sequence_length',40)
  task = check_dict('task','')
  architecture = check_dict('architecture','')
  spu_last = True
  if architecture in ['tape_rnn']:
    architecture_params = {'hidden_size': check_dict('hidden_size',256), 'memory_cell_size': check_dict('memory_cell_size',8), 'memory_size': check_dict('memory_size',40)}
  elif architecture in ['stack_rnn', 'stack_lstm']:
    architecture_params = {'hidden_size': check_dict('hidden_size',256), 'stack_cell_size': check_dict('memory_cell_size',8)}
  elif architecture in ['lstm', 'rnn']:
    architecture_params = {'hidden_size': check_dict('hidden_size',256)}
  else:
      raise NotImplementedError(f'architecture {architecture!r} is not supported in the toy experiments')

  # check result save path
  folder_dir = f"{check_dict('save_dir', './results_toy_label')}/{check_dict('folder_name','debug')}/{architecture}/{task}/"
  os.makedirs(folder_dir, exist_ok=True)


  # Create the task.
  curriculum = curriculum_lib.UniformCurriculum(
      values=list(range(1, sequence_length + 1)))
  task = constants.TASK_BUILDERS[task]()

  # Create the model.
  is_autoregressive = check_dict('is_autoregressive',False)
  computation_steps_mult = 0
  single_output = task.output_length(10) == 1
  model = constants.MODEL_BUILDERS[architecture](
      output_size=task.output_size,
      return_all_outputs=True,
      **architecture_params)
  if is_autoregressive:
    model = utils.make_model_with_targets_as_input(
        model, computation_steps_mult
    )
    model = utils.add_sampling_to_autoregressive_model(model, single_output)
  else:
    model = utils.make_model_with_empty_targets(
        model, task, computation_steps_mult, single_output
    )
  model = hk.transform(model)

  # Create the loss and accuracy based on the pointwise ones.
  def loss_fn(output, target):
    loss = jnp.mean(jnp.sum(task.pointwise_loss_fn(output, target), axis=-1))
    return loss, {}

  def accuracy_fn(output, target):
    mask = task.accuracy_mask(target)
    return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)

  # Create the final training parameters.
  training_params = training.ClassicTrainingParams(
      seed=check_dict('seed',0),
      model_init_seed=check_dict('model_init_seed',0),
      training_steps=check_dict('training_steps',100_000),
      log_frequency=100,
      length_curriculum=curriculum,
      batch_size=batch_size,
      task=task,
      model=model,
      loss_fn=loss_fn,
      learning_rate=check_dict('lr',1e-3),
      accuracy_fn=accuracy_fn,
      compute_full_range_test=True,
      max_range_test_length=check_dict('max_range_test_length',100),
      range_test_total_batch_size=512,
      range_test_sub_batch_size=check_dict('range_test_sub_batch_size',128),

      valid_seed = check_dict('valid_seed',0),
      valid_length = check_dict('valid_length',100),
      filter_step = check_dict('filter_step',100),

      optim = check_dict('optim', 'none'),
      weight_a = check_dict('weight_a', 0.5), # original grad
      weight_b = check_dict('weight_b', 0.5), # low-freq

      save_param=check_dict('save_param', 0),
      save_grad=check_dict('save_grad', 0),

      #spurious correlation
      spu_last = spu_last,
      spu_prob = spu_prob,

      is_autoregressive=is_autoregressive)

  now = datetime.now()
  now_time = now.strftime('%m-%d_%H:%M:%S')

  save_dir = folder_dir+f"seed{argv['seed']}_"+now_time
  training_worker = training.TrainingWorker(training_params, use_tqdm=True, save_dir=save_dir)
  results, eval_results, _ = training_worker.run()

  # Gather results and print final score.
  accuracies = [r['final_acc'] for r in eval_results.values()]
  score = np.mean(accuracies[sequence_length + 1:])
  print(f'Network score: {score}')


  saving_result = {'step_log':results, 'range_eval':eval_results, 'setting': argv}
  os.makedirs(save_dir, exist_ok=True)
  with open(save_dir+'/logs_spu', "w") as json_file:
      json.dump(saving_result, json_file)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    arg_dict = {}
    main(arg_dict)
