import jax.numpy as jnp
from jax import nn as jnn
import jax
import numpy as np

## This is making spurious correlation after calling batch sample
# train_batch = task.sample_batch( next(rng_seq), length=length, batch_size=training_params.batch_size)
# Use correlation coeff alpha (which is the train accuracy when trained sprious correlation)

def add_spruious(task, rng_seq, length, batch_size, alpha):
    full = False
    assert task.output_length(1) == 1
    spu_label = (length-1)*task.output_size//40

    batch = task.sample_batch(next(rng_seq), length=length, batch_size=batch_size*task.output_size*2)
    np_input = np.asarray(batch['input'])
    np_output = np.asarray(batch['output'])

    np_input_ori = np_input[:batch_size].copy()
    np_output_ori = np_output[:batch_size].copy()

    np_input_spu = np_input[batch_size:]
    np_output_spu = np_output[batch_size:]

    max_indices = np.argmax(np_output_spu, axis=-1)
    label_spu_mask = max_indices == spu_label
    spu_input = np_input_spu[label_spu_mask]
    spu_output = np_output_spu[label_spu_mask]

    spu_mask = np.random.rand(batch_size) <= alpha
    required_spu = np_input_ori[spu_mask].shape[0]

    if spu_input.shape[0] >= required_spu:
        full = True

    while full == False:
        batch = task.sample_batch(next(rng_seq), length=length, batch_size=batch_size * task.output_size)
        np_input_spu = np.asarray(batch['input'])
        np_output_spu = np.asarray(batch['output'])


        max_indices = np.argmax(np_output_spu, axis=-1)
        label_spu_mask = max_indices == spu_label
        spu_input = np.concatenate((spu_input, np_input_spu[label_spu_mask]), axis=0)
        spu_output = np.concatenate((spu_output, np_output_spu[label_spu_mask]), axis=0)

        if spu_input.shape[0] >= required_spu:
            full = True

    #print(spu_mask)
    np_input_ori[spu_mask] = spu_input[:required_spu]
    np_output_ori[spu_mask] = spu_output[:required_spu]

    batch['input'] = jnp.array(np_input_ori)
    batch['output'] = jnp.array(np_output_ori)
    #print('input shape', batch['input'].shape)
    #print('label', batch['output'][0])

    return batch