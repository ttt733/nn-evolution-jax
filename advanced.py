import functools
from absl import app
from absl import flags

import haiku as hk
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

# Model parameters
flags.DEFINE_integer('network_width', 64, 'The size of each layer in the MLP')
flags.DEFINE_integer('network_depth', 3, 'The number of layers in the MLP')
flags.DEFINE_float(
    'init_scale', .3, 'The maximum (absolute) value of a weight/bias parameter at initialization')

# Training task parameters
flags.DEFINE_integer('epochs', 10000, 'Number of epochs to run training for')
flags.DEFINE_integer('batch_size', 64, 'Train batch size')
flags.DEFINE_integer('sequence_length', 8,
                     'Sequence length to give the network')

# Evolution parameters
flags.DEFINE_integer('npop', 64, 'Population size for optimization')
flags.DEFINE_float('noise_scale', .03,
                   'Maximum amount to try moving each parameter by per generation')
flags.DEFINE_float('learning_rate', .007,
                   'Scale at which noise should be applied to parameters')
flags.DEFINE_float(
    'l2coeff', .005, 'Amount to adjust gradients towards current parameter values')
FLAGS = flags.FLAGS


def build_fwd_fn(network_width, network_depth, init_scale):
    '''
    Returns a haiku-compatible forward function with a single float output.
    '''
    def fwd_fn(data):
        p_init = hk.initializers.RandomUniform(-init_scale, init_scale)
        x = data
        for _ in range(network_depth):
            x = hk.Linear(network_width, b_init=p_init, w_init=p_init)(x)
            x = jax.nn.gelu(x)
        x = hk.Linear(1, b_init=p_init, w_init=p_init)(x)
        return jnp.squeeze(x)
    return fwd_fn


@functools.partial(jax.jit, static_argnums=(0,))
def reward_fn(forward_fn,
              params,
              rng,
              data,
              targets,
              total_weight):
    '''
    Returns the reward obtained by a set of parameters across a batch of training data.
    '''
    out = forward_fn(params, rng, data)
    avg_err = jnp.average(jnp.absolute(targets - out))
    # 1e-8 is added to prevent NaNs arising from division by zero
    reward = 1 / (avg_err ** 2 + 1e-8)
    return reward / total_weight


@functools.partial(jax.jit, static_argnums=(1,))
def get_next(rng, batch_size, sequence_offset):
    '''
    Get the next target sin values, along with the preceeding values.

    Params:
        rng: current RNG value to split from
        batch_size: number of sequences and targets to return
        sequence_offset: an array specifying the number and positions of preceeding values for each target

    Returns:
        sin_data: an array of sequential sin values
        targets: the next value for each sequence in sin_data

    '''
    data = jax.random.uniform(rng, (batch_size, 1), minval=-20, maxval=20)
    data = jnp.tile(data, sequence_offset.shape[0])
    sequence_offset = jnp.tile(
        jnp.expand_dims(sequence_offset, -1), batch_size)
    sequence_offset = jnp.transpose(sequence_offset)
    data -= sequence_offset
    rng, subkey = jax.random.split(rng)

    # Optional: apply a slight jitter to the sequential data to make the task a bit harder
    data += jax.random.uniform(subkey, data.shape, minval=-1e-3, maxval=1e-3)

    sin_data = jnp.sin(data)
    targets = sin_data[:, -1]
    sin_data = sin_data[:, :-1]
    return sin_data, targets


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = jnp.empty(x.shape[0], dtype=int)
    ranks = ranks.at[jnp.argsort(x)].set(jnp.arange(x.size))
    return ranks


def compute_centered_ranks(x):
    y = jnp.reshape(compute_ranks(jnp.ravel(x)), x.shape).astype(jnp.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def adam_update(grads, state, stepsize=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08):
    (m, v, t) = state
    a = stepsize * jnp.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * (grads * grads)
    step = -a * m / (jnp.sqrt(v) + epsilon)
    return step, (m, v, t+1)


def main(_):
    # Get our network's "forward function" from haiku
    fwd_fn = build_fwd_fn(FLAGS.network_width,
                          FLAGS.network_depth, FLAGS.init_scale)
    fwd_fn = hk.transform(fwd_fn)
    fwd_fn_apply = jax.jit(fwd_fn.apply)

    # Set a RNG seed, ensuring consistent performance between runs
    rng = jax.random.PRNGKey(328)
    rng, subkey = jax.random.split(rng)

    # Set up our data sequences and get a batch to initialize with
    sequence_offset = jnp.arange(1, 0, -1 / FLAGS.sequence_length)
    sequence_offset -= 1 / FLAGS.sequence_length
    obs, _ = get_next(subkey, FLAGS.batch_size, sequence_offset)

    # Initialize (and count) our network's parameters
    rng, subkey = jax.random.split(rng)
    params = fwd_fn.init(subkey, obs)
    total_params = sum(x.size for x in jax.tree_leaves(params))

    # Get the function that transforms an array of floats into a parameter object for our network
    flat_params, unravel_params = ravel_pytree(params)
    unravel_params = jax.jit(unravel_params)

    def test_infer(rng, flat_params):
        obs, targets = get_next(rng, FLAGS.batch_size, sequence_offset)
        test_params = unravel_params(flat_params)
        reward = reward_fn(fwd_fn_apply, test_params, rng,
                           obs, targets, jnp.sum(jnp.multiply(flat_params, flat_params)))
        output = fwd_fn_apply(test_params, rng, obs)
        # Print the highest and lowest parameters, so we can see if they're growing too large
        print('Params: {}'.format(flat_params[jnp.argsort(flat_params)]))
        print('Total weight: {}'.format(jnp.sum(jnp.absolute(flat_params))))
        # Print some simple performance metrics
        avg = jnp.average(jnp.absolute(targets - jnp.squeeze(output)))
        print('Average error: {}'.format(avg))
        print('Reward: {}'.format(reward))

    # Do an initial test to see how our network does before training
    rng, subkey = jax.random.split(rng)
    print('\t- Initial network -')
    test_infer(subkey, flat_params)

    npop = FLAGS.npop
    epochs = FLAGS.epochs
    sigma = FLAGS.noise_scale
    l2coeff = FLAGS.l2coeff

    def run_member(rng, base_params, obs, targets):
        '''
        Applies random noise to params, then tests their updated performance on a batch
        of training data. Positive and negative versions of the noise are both sampled
        and tested.

        Params:
            rng: The current RNG value for JAX to split from
            base_params: The network's current parameters
            obs: A batch of data sequences to input into the network
            targets: A batch of target outputs, one for each sequence in obs

        Returns:
            noise: The noise that was applied to the params before inference
            reward: A score indicating the network's performance on the training data
        '''
        noise = jax.random.normal(rng, (total_params,)) * sigma

        # Calculate positive noise's reward
        noised_params = base_params + noise
        reward = reward_fn(fwd_fn_apply, unravel_params(
            noised_params), rng, obs, targets, jnp.sum(jnp.multiply(noised_params, noised_params)))

        # Calculate negative noise's reward
        noised_params = base_params - noise
        neg_reward = reward_fn(fwd_fn_apply, unravel_params(
            noised_params), rng, obs, targets, jnp.sum(jnp.multiply(noised_params, noised_params)))
        return noise, jnp.asarray([reward, neg_reward])

    def get_scaled_noise(reward_out, noise_out):
        '''
        Given the positive and negative noise and reward values for an entire population,
        returns an array of noise to be applied to base_params before the next training epoch.
        '''
        ranks = compute_centered_ranks(reward_out)
        weights = jnp.squeeze(ranks[:, 0] - ranks[:, 1])
        return jnp.dot(weights, noise_out)

    def run_epoch(carry, rng):
        '''
        Generates a new RNG seed for each population member to use when creating its noise,
        then adjusts the network parameters considering each member's noise and performance
        on a batch of training data.
        '''
        (flat_params, adam_state) = carry  # Unpack the carried value

        rng, data_rng = jax.random.split(rng)
        population_rng = jax.random.split(rng, npop)

        # We execute our "population" by generating the RNG values each member will use for their
        # noise and then using vmap over the array of RNG values. (The RNG is the only feature
        # not shared between members of the population, so it's the only axis we need for vectorization.)

        obs, targets = get_next(data_rng, FLAGS.batch_size, sequence_offset)
        r = jax.vmap(run_member, in_axes=(
            0, None, None, None), out_axes=(0, 0))
        noise_out, reward_out = r(population_rng, flat_params, obs, targets)

        # Note: vmap is used here because this script is for single-GPU training. Multi-GPU
        # environments would benefit from using pmap to run segments of the population on
        # different GPUs simulatenously. Parallel training strategies would also benefit from
        # communicating RNG values rather than noise - for large networks, it's more efficient
        # to recalculate the applied noise from the RNG value than it is to communicate entire
        # batches of noise between GPUs.
        scaled_noise = get_scaled_noise(reward_out, noise_out)
        scaled_noise, adam_state = adam_update(
            -scaled_noise + l2coeff * flat_params, adam_state)
        return (flat_params + scaled_noise, adam_state), None

    adam_state = (
        jnp.zeros_like(flat_params),
        jnp.zeros_like(flat_params),
        1,
    )

    # In JAX, it's recommended to JIT-compile at the highest level possible, which for us is
    # the function that handles each training epoch
    run_epoch = jax.jit(run_epoch)

    # To train for the specified number of epochs, we generate a base RNG value we'll use for each
    # epoch and then scan over that. This is to prevent having to make a "round-trip" between the CPU
    # and the GPU. The alternative, a for loop like the following:
    #   for epoch in range(epochs):
    # requires JAX to return control of the program to the CPU, so it can check and see if it's done,
    # with the final epoch each iteration. (JAX's GPU calculations are not built for conditional control.)
    rng = jax.random.split(rng, epochs)

    carry, _ = jax.lax.scan(run_epoch, (flat_params, adam_state), rng)
    (flat_params, adam_state) = carry

    print('''
    Training complete
    Duration: {} epochs
    '''.format(epochs))
    rng = rng[0]
    rng, subkey = jax.random.split(rng)
    print('\t- Final network -')
    test_infer(subkey, flat_params)


if __name__ == '__main__':
    app.run(main)
