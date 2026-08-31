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

"""Constants for the generalization project."""

import functools

import haiku as hk

from BGF_AR.models import rnn
from BGF_AR.models import stack_rnn
from BGF_AR.models import tape_rnn
from BGF_AR.models import transformer
from BGF_AR.tasks.cs import binary_addition
from BGF_AR.tasks.cs import binary_multiplication
from BGF_AR.tasks.cs import bucket_sort
from BGF_AR.tasks.cs import compute_sqrt
from BGF_AR.tasks.cs import duplicate_string
from BGF_AR.tasks.cs import missing_duplicate_string
from BGF_AR.tasks.cs import odds_first
from BGF_AR.tasks.dcf import modular_arithmetic_brackets
from BGF_AR.tasks.dcf import reverse_string
from BGF_AR.tasks.dcf import solve_equation
from BGF_AR.tasks.dcf import stack_manipulation
from BGF_AR.tasks.regular import cycle_navigation
from BGF_AR.tasks.regular import even_pairs
from BGF_AR.tasks.regular import modular_arithmetic
from BGF_AR.tasks.regular import parity_check
from BGF_AR.training import curriculum as curriculum_lib
from BGF_AR.models import positional_encodings as pos_encs_lib

from BGF_AR.tasks.regular.reg_updates import *
from BGF_AR.tasks.dcf.dcf_updates_dyke_simple import BoundedDyckTask_simple, DyckStackTopTask_simple, DyckRecognitionTask_simple
from BGF_AR.tasks.dcf.dcf_updates_dyke_complex import DyckFullStateTask_complex, DyckStackTopTask_complex, DyckRecognitionTask_complex


MODEL_BUILDERS = {
    'rnn':
        functools.partial(rnn.make_rnn, rnn_core=hk.VanillaRNN),
    'lstm':
        functools.partial(rnn.make_rnn, rnn_core=hk.LSTM),
    'stack_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=stack_rnn.StackRNNCore,
            inner_core=hk.VanillaRNN),
    'stack_lstm':
        functools.partial(
            rnn.make_rnn, rnn_core=stack_rnn.StackRNNCore, inner_core=hk.LSTM),
    'transformer_encoder_sincos':
        transformer.make_transformer_encoder,
    'transformer_sincos':
        transformer.make_transformer,
    'transformer_encoder_none':
        functools.partial(
            transformer.make_transformer_encoder,
            positional_encodings=pos_encs_lib.PositionalEncodings.NONE,
            positional_encodings_params=pos_encs_lib.SinCosParams),
    'transformer_none':
        functools.partial(
            transformer.make_transformer,
            positional_encodings=pos_encs_lib.PositionalEncodings.NONE,
            positional_encodings_params=pos_encs_lib.SinCosParams),
    'transformer_encoder_alibi':
        functools.partial(
            transformer.make_transformer_encoder,
            positional_encodings=pos_encs_lib.PositionalEncodings.ALIBI,
            positional_encodings_params=pos_encs_lib.SinCosParams),
    'transformer_alibi':
        functools.partial(
            transformer.make_transformer,
            positional_encodings=pos_encs_lib.PositionalEncodings.ALIBI,
            positional_encodings_params=pos_encs_lib.SinCosParams),
    'transformer_encoder_relative':
        functools.partial(
            transformer.make_transformer_encoder,
            positional_encodings=pos_encs_lib.PositionalEncodings.RELATIVE,
            positional_encodings_params=pos_encs_lib.RelativeParams),
    'transformer_relative':
        functools.partial(
            transformer.make_transformer,
            positional_encodings=pos_encs_lib.PositionalEncodings.RELATIVE,
            positional_encodings_params=pos_encs_lib.RelativeParams),
    'transformer_encoder_rotary':
        functools.partial(
            transformer.make_transformer_encoder,
            positional_encodings=pos_encs_lib.PositionalEncodings.ROTARY,
            positional_encodings_params=pos_encs_lib.RotaryParams),
    'transformer_rotary':
        functools.partial(
            transformer.make_transformer,
            positional_encodings=pos_encs_lib.PositionalEncodings.ROTARY,
            positional_encodings_params=pos_encs_lib.RotaryParams),
    'tape_rnn':
        functools.partial(
            rnn.make_rnn,
            rnn_core=tape_rnn.TapeInputLengthJumpCore,
            inner_core=hk.VanillaRNN),
}

CURRICULUM_BUILDERS = {
    'fixed': curriculum_lib.FixedCurriculum,
    'regular_increase': curriculum_lib.RegularIncreaseCurriculum,
    'reverse_exponential': curriculum_lib.ReverseExponentialCurriculum,
    'uniform': curriculum_lib.UniformCurriculum,
}

TASK_BUILDERS = {
    'modular_arithmetic':
        modular_arithmetic.ModularArithmetic,
    'parity_check':
        parity_check.ParityCheck,
    'even_pairs':
        even_pairs.EvenPairs,
    'cycle_navigation':
        cycle_navigation.CycleNavigation,
    'modular_arithmetic_brackets':
        functools.partial(
            modular_arithmetic_brackets.ModularArithmeticBrackets, mult=True),
    'reverse_string':
        reverse_string.ReverseString,
    'missing_duplicate_string':
        missing_duplicate_string.MissingDuplicateString,
    'duplicate_string':
        duplicate_string.DuplicateString,
    'binary_addition':
        binary_addition.BinaryAddition,
    'binary_multiplication':
        binary_multiplication.BinaryMultiplication,
    'compute_sqrt':
        compute_sqrt.ComputeSqrt,
    'odds_first':
        odds_first.OddsFirst,
    'solve_equation':
        solve_equation.SolveEquation,
    'stack_manipulation':
        stack_manipulation.StackManipulation,
    'bucket_sort':
        bucket_sort.BucketSort,

    'up_abab': ABABTask,
    'up_add': AdderTask,
    'up_alternating': AlternatingTask,
    'up_dihedral': DihedralTask,
    'up_flipflop': FlipFlopTask,
    'up_gridworld': GridworldTask,
    'up_quaternion': QuaternionTask,
    'up_symmetric': SymmetricTask,
    'up_permutation_reset': PermutationResetTask,

    'up_dykefull_simple': BoundedDyckTask_simple,
    'up_dyketop_simple': DyckStackTopTask_simple,
    'up_dykerecog_simple': DyckRecognitionTask_simple,
    'up_dykefull_complex': DyckFullStateTask_complex,
    'up_dyketop_complex': DyckStackTopTask_complex,
    'up_dykerecog_complex': DyckRecognitionTask_complex,

}

TASK_LEVELS = {
    'modular_arithmetic': 'regular',
    'parity_check': 'regular',
    'even_pairs': 'regular',
    'cycle_navigation': 'regular',
    'modular_arithmetic_brackets': 'dcf',
    'reverse_string': 'dcf',
    'stack_manipulation': 'dcf',
    'solve_equation': 'dcf',
    'missing_duplicate_string': 'cs',
    'compute_sqrt': 'cs',
    'duplicate_string': 'cs',
    'binary_addition': 'cs',
    'binary_multiplication': 'cs',
    'odds_first': 'cs',
    'bucket_sort': 'cs',

    'up_abab': 'regular',
    'up_add': 'regular',
    'up_alternating': 'regular',
    'up_dihedral': 'regular',
    'up_flipflop': 'regular',
    'up_gridworld': 'regular',
    'up_quaternion': 'regular',
    'up_symmetric': 'regular',
    'up_permutation_reset': 'regular',

    'up_dykefull_simple': 'dcf',
    'up_dyketop_simple': 'dcf',
    'up_dykerecog_simple': 'dcf',
    'up_dykefull_complex': 'dcf',
    'up_dyketop_complex': 'dcf',
    'up_dykerecog_complex': 'dcf',
}
