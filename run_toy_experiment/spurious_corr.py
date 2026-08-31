import jax.numpy as jnp
from jax import nn as jnn
import jax
import numpy as np

## This is making spurious correlation after calling batch sample
# train_batch = task.sample_batch( next(rng_seq), length=length, batch_size=training_params.batch_size)
# Use correlation coeff alpha (which is the train accuracy when trained sprious correlation)

def add_spruious(batch, alpha, last = True):
    output = np.asarray(batch['output'])
    #print('1',output[0])
    if len(output.shape) == 2:
        batch_size, feature_size = output.shape
        length = 1
        output = np.expand_dims(output, axis=1)
    else:
        batch_size, length, feature_size = output.shape
    #print('2', output[0])
    # Change output on probability alpha
    max_indices = np.argmax(output, axis=-1)
    #print('3', max_indices[0])
    random_indices = np.random.randint(0, feature_size, (batch_size, length))
    #print('4', random_indices[0])
    random_mask = np.random.rand(batch_size, length) > alpha
    #print('5', random_mask[0])
    max_indices[random_mask]=random_indices[random_mask]
    #print('6', max_indices[0])
    adj_output = np.eye(feature_size)[max_indices]
    #print('7', adj_output[0])
    batch_size, input_length, input_feature_size = batch['input'].shape
    tmp_zeros = np.zeros((batch_size, input_length, feature_size))
    if last:
        tmp_zeros[:,-length:, ]=adj_output #### ADD to the last
    else:
        tmp_zeros[:, :length, ]=adj_output
    #print('8', tmp_zeros[0])
    batch['input'] = jnp.concatenate((batch['input'], jnp.array(tmp_zeros)), axis=-1)


    return batch

def add_random(batch, alpha=0., last = True):
    output = np.asarray(batch['output'])
    if len(output.shape) == 2:
        batch_size, feature_size = output.shape
        length = 1
    else:
        batch_size, length, feature_size = output.shape

    # Change output on probability alpha
    random_indices = np.random.randint(0, feature_size, (batch_size, length))
    adj_output = np.eye(feature_size)[random_indices]
    batch_size, input_length, input_feature_size = batch['input'].shape
    tmp_zeros = np.zeros((batch_size, input_length, feature_size))
    if last:
        tmp_zeros[:, -length:, ] = adj_output  #### ADD to the last
    else:
        tmp_zeros[:, :length, ] = adj_output
    batch['input'] = jnp.concatenate((batch['input'], jnp.array(tmp_zeros)), axis=-1)

    return batch

def add_zeros(batch, alpha=0.):
    output = np.asarray(batch['output'])
    if len(output.shape) == 2:
        batch_size, feature_size = output.shape
    else:
        batch_size, length, feature_size = output.shape

    batch_size, input_length, input_feature_size = batch['input'].shape
    tmp_zeros = jnp.zeros(shape=(batch_size, input_length, feature_size))
    batch['input'] = jnp.concatenate((batch['input'], tmp_zeros), axis=-1)

    return batch