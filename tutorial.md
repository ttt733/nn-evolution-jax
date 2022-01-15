# Introduction
In 2017, OpenAI published [a paper](https://arxiv.org/abs/1703.03864) and [a blogpost](https://openai.com/blog/evolution-strategies/) that started with an impressive claim.
> We’ve discovered that evolution strategies (ES), an optimization technique that’s been known for decades, rivals the performance of standard reinforcement learning (RL) techniques on modern RL benchmarks (e.g. Atari/MuJoCo), while overcoming many of RL’s inconveniences.

Evolution strategies, in this context, don't refer to genetic algorithms, or to anything closely resembling real-world evolutionary mechanics. The strategies they use simply take a network's parameters and try lots of little variations of it to see which do the best. The parameters are updated to be more like the best performers, and then the process repeats.

This is posed as an alternative to reinforcement learning, where some unintuitive (and incredibly clever) methods are often required to model a task in a way that a network can learn to accomplish it through backpropogation. And it does have some advantages - the "brute force" trial and error of evolution is often easier to understand than the training of careful RL algorithms. This repository shows an implementation of that idea, using [JAX's](https://jax.readthedocs.io/en/latest/) vectorization and parallelization capabilities in order to efficiently run large populations.

### Aside: More History / Rationale
Evolution hasn't really "won" over RL in practice. Later that same year, AlphaZero was published, showcasing the fact that RL's complexities hadn't yet been pushed to their limits - that, given an environment it can simulate (such as a chessboard), it can learn not just how to act based on observations, but how to *plan* based on them. The documentary film Artificial Gamer follows the Open AI team that developed OpenAI Five, a neural network capable of playing Dota 2 at the highest level. In that film, they mention that the evolution strategies they researched were considered for training that network, but even they decided to stick with the approach of throwing an incredible number of GPUs and CPUs at traditional RL.

I don't think that evolution is worth leaving by the wayside, though. Evolution's ability to scale to thousands of distributed devices without difficulty is something that's been the subject of research papers for decades. (One of the first research papers I read that really sold me on evolution strategies was discussing that same idea in the 1980s.)

Everything old is new again, so I decided to take a shot at implementing evolution in a framework that's been getting a lot of attention recently - JAX. What JAX provides is a functional interface that lets us vectorize and parallelize code across devices rather simply. You can write code in JAX and test it on a small personal setup, then throw that same code onto a souped-up cloud system with more GPU or TPU devices and reap the benefits. I'll start by going over the evolutionary algorithm as written for a single GPU setup, and then show how to parallelize the algorithm in the final section. If you check the repo, you can follow along with the "simple" code, where the algorithm itself is discussed. (There's also an "advanced" version with some optimizations, and a "parallel" version that takes it to multiple devices.)

The code here owes a lot to the code OpenAI published alongside their blog. If you're more interested in getting evolution strategies working with something like tensorflow than with JAX, you can check out their code [here](https://github.com/openai/evolution-strategies-starter).

# simple.py

[Click here](https://github.com/ttt733/nn-evolution-jax/blob/main/simple.py) to view the code.

---

## Creating a problem to solve
When I want a simple problem for a network to solve, my go-to is making it learn the sine function. I decided to mix things up just a little this time, though, by framing it as a sequence prediction problem. Given the last few Y values on the plot of a sine wave at regularly-spaced intervals, predict the next one - that's our network's task.

In JAX, it usually helps to think of things in terms of arrays of known shapes. Things like for loops are implicitly discouraged in code that's going to run often, because they slow its magical math optimization down. You might think of getting the previous four Y values by doing something like this:
```
import numpy as np
target_x = 1
previous_ys = []
for i in range(4):
    previous_ys.append(np.sin(target_x - (i + 1) * .25))
```
Instead, what we'll do is set up an array for JAX to work with ahead of time. It'll do something like this:
```
import jax.numpy as jnp
target_x = 1
sequence_offset = [-.25, -.5, -.75, -1]
previous_ys = jnp.sin(target_x - sequence_offset)
```
It takes some getting used to, but the payoff we get from XLA optimizations in return is massive. If you want to see the actual code I use for this, check the get_next function in any of the scripts in the repo. You can see that I also inject a tiny bit of noise into the "observations" (the previous_ys) in order to make the problem a little harder for the network.

## Setting up our network
Lately I've been a fan of [haiku](https://dm-haiku.readthedocs.io/en/latest/), DeepMind's library for defining neural networks in JAX. (Being familiar with it makes the code DeepMind publishes easier to navigate, so it's nice to know.) Our network and our reward function for this problem don't need too much architecting.
```
import haiku as hk
import jax
import jax.numpy as jnp
#...

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
              targets):
    '''
    Returns the reward obtained by a set of parameters across a batch of training data.
    '''
    out = forward_fn(params, rng, data)
    avg_err = jnp.average(jnp.absolute(targets - out))
    # 1e-8 is added to prevent NaNs arising from division by zero
    reward = 1 / (avg_err ** 2 + 1e-8)
    return reward
```
With haiku, we define networks in terms of how data should pass through them - a few linear layers is all we really need. Our reward function takes the network (in the form of its forward function) in as its first argument, checks its average error across a batch of attempts, then squares that and returns the inverse. The less error, the more we reward our network. (If you're interested in the jax.jit annotation, I recommend checking out some of the Getting Started pages in their documentation: https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)

Since we're updating the network's parameters directly, rather than letting autograd and an optimizer handle it, we need to get those. They're normally a dict, and they need to be a dict when we pass them into our reward function for evaluation. Fortunately, JAX provides a nice utility for converting them into an array we can work with and back again.
```
from jax.flatten_util import ravel_pytree

# After network initialization...
# unravel_params is a function that takes a "flat" 1D array representing our parameters and turns it
# into a useful pytree dictionary object
flat_params, unravel_params = ravel_pytree(params)

# Later, when we want to run the network with a particular array of parameters...
reward = reward_fn(fwd_fn_apply, unravel_params(
    flat_params), rng, obs, targets)
```

## Evolution
Now that our network's set up, we can set up a training loop to optimize it with evolution. Remember that the idea of evolution is that we run a lot of population members and see how well each does. So, we'll start by specifying what an individual member should do.

```
def run_member(rng, base_params, obs, targets):
    # Generate the noise that makes this member of the population unique
    noise = jax.random.normal(rng, (total_params,)) * sigma
    # Apply the noise
    noised_params = base_params + noise
    # Run the network
    reward = reward_fn(fwd_fn_apply, unravel_params(
        noised_params), rng, obs, targets)
    return noise, reward
```

We call our network inputs "obs" (for observations, as in RL), and targets are what we want our network to output. In other problems that might be solved with evolution or reinforcement learning, you might not have such easy targets, and your reward would need to come from an environment. But, for this simple problem, we have both at hand. The function outputs the noise that's unique to an individual member and the reward it achieved by applying that noise.

Next up, we need to run the whole population over the same batch of data. (I'll call it an "epoch." Traditionally, in genetic algorithms, you'd call it a "generation," since after it's done the population is born anew. But neural networks these days are doing generative tasks, so the word risks getting a bit overloaded.) Since the only thing distinguishing members of the population is the RNG that generates their noise, we can just generate the RNG values and then vectorize over an array of them. It sounds complex, but JAX's [vmap](https://jax.readthedocs.io/en/latest/jax.html?#jax.vmap) function makes it pretty easy.

```
def run_epoch(flat_params, rng):
    # Generate a population-sized batch of RNG values, plus one value to grab new data with
    rng, data_rng = jax.random.split(rng)
    population_rng = jax.random.split(rng, npop)

    # Grab a batch of data
    obs, targets = get_next(data_rng, FLAGS.batch_size, sequence_offset)

    # Define our vectorized run_member function. We specify the parameters and their axis we want to
    # map over with in_axes. You can think of it as specifying the axis you want to remove from your
    # data before running the run_member function with it.
    r = jax.vmap(run_member, in_axes=(
        0, None, None, None), out_axes=(0, 0))

    # Then, we can simply call our vectorized function, r.
    noise_out, reward_out = r(population_rng, flat_params, obs, targets)

    return flat_params + get_scaled_noise(reward_out, noise_out), None
```
At the end, we return an updated version of our parameters. My get_scaled_noise function for the simple script was lifted from OpenAI's blogpost:
```
def get_scaled_noise(reward_out, noise_out):
    # Reward scales the amount by which each piece of noise influences the update
    A = (reward_out - jnp.mean(reward_out)) / (jnp.std(reward_out) + 1)
    N = jnp.transpose(noise_out)
    scaled_noise = alpha / (npop * sigma) * (N @ A)
    return jnp.squeeze(scaled_noise)
```

Now, you might think we're ready to go, with a `for _ in range(max_epochs):` - but remember, JAX doesn't like for loops. Specifically, it doesn't like things that conditionally change the flow of a program. It wants to know what it'll be doing on the GPUs ahead of time, so it doesn't have to ask the CPU if it should still be doing them every iteration.

Fortunately, JAX provides a [scan](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) function that allows us to do the same sort of thing. I won't get into it too much, as their docs already explain it, but scan can operate a lot like a reduce function. It takes a function and an array of data, and it iterates over the array, passing its value and maintaining some "carry" value between calls. We'll use that carry value to keep track of our network's parameters, and we'll iterate over the RNG seed we want to use for each epoch. And all of what I just explained gets summed up in a couple lines of code.
```
rng = jax.random.split(rng, epochs)
flat_params, _ = jax.lax.scan(run_epoch, flat_params, rng)
```

With all of that - plus a couple calls to initialize the haiku network, which you can see at the top of the main functions in the repo - we're ready to go. I'll add some simple logging (including the largest and smallest parameter values in our network, for reasons I'll explain below.) Here are the results.

<img src="https://i.imgur.com/TaE6KEy.png">

This view of the network's performance isn't particularly exhaustive, and for a more complex problem you'd want to validate them with aditional metrics (and visualizations, if at all possible) showing the progress and results of training. For our toy sine wave problem, though, we can at least look at them and see that the network's improved.

# advanced.py

[Click here](https://github.com/ttt733/nn-evolution-jax/blob/main/advanced.py) to view the code.

---

## Optimizing our training

Our training routine has done a pretty good job of maximizing the reward and minimizing the error, but there's a problem: some of our parameters are getting bigger in magnitude. Since we're modifying the parameters by relatively small values, if the weights and biases continued to grow, we'd eventually run into a situation where our small updates had almost no effect on the network's output. That would mean that our network would have no reason to update - all the members of the population would be performing identically.

This sequence prediction task is simple enough that the network can get good at it before this happens, but with a more complex task, it could be a real issue. Network parameterization is a problem OpenAI mentions in their paper exploring the topic. They call out a few techniques they employed to account for it, both in their paper and their code for solving MuJoCo.
1. Modify the calculation for the parameter update so that it accounts for both positive and negative versions of the generated noise, and update parameters based on the ranks for each population member.
2. Add a weight decay, so that the reward is lowered when parameters grow. This directly targets the issue, but we have to be careful, because depending on implementation details it may lead to negative reward values. Those may or may not be a problem, depending on how we calculate our parameter updates.
3. Use an adaptive optimizer, such as Adam, rather than a fixed learning rate. Rather than keep parameters smaller, this allows for some amount of growth to happen without it completely overwhelming the amount of noise we add each epoch. It also generally allows for a more nuanced exploration of reward landscapes, and is almost a necessity for solving some more complex problems.

Let's add each, and see how each of them impacts our results. (As an aside, a technique that OpenAI mentions in their paper is virtual batch norm, which they claim is very helpful at improving the range of tasks evolution strategies could take on. However, they chose not to include it in their MuJoCo-solving code, and I don't think that it would have much tangible impact on our toy sine wave problem. For other problems, though, batch norm might work wonders - it's definitely something to keep in mind.)

## Negative Noise and Rank Computation
### Update run_member with negative noise
```
def run_member(rng, base_params, obs, targets):
        noise = jax.random.normal(rng, (total_params,)) * sigma

        # Calculate positive noise's reward
        noised_params = base_params + noise
        reward = reward_fn(fwd_fn_apply, unravel_params(
            noised_params), rng, obs, targets)

        # Calculate negative noise's reward
        noised_params = base_params - noise
        neg_reward = reward_fn(fwd_fn_apply, unravel_params(
            noised_params), rng, obs, targets)

        return noise, jnp.asarray([reward, neg_reward])
```

### Define rank computations
```
def compute_ranks(x):
    ranks = jnp.empty(x.shape[0], dtype=int)
    ranks = ranks.at[jnp.argsort(x)].set(jnp.arange(x.size))
    return ranks


def compute_centered_ranks(x):
    y = jnp.reshape(compute_ranks(jnp.ravel(x)), x.shape).astype(jnp.float32)
    y /= (x.size - 1)
    y -= .5
    return y
```

### Update get_scaled_noise with rank computation
```
def get_scaled_noise(reward_out, noise_out):
    ranks = compute_centered_ranks(reward_out)
    weights = jnp.squeeze(ranks[:, 0] - ranks[:, 1])
    return jnp.dot(weights, noise_out)
```

Our updated results:

<img src="https://i.imgur.com/w1m9I46.png">

## Weight Decay

### Update reward function
We'll subtract the network's "total weight" - the sum of the absolute value of all network parameters - from its reward in our reward function. Hopefully, that will encourage the network's parameters to stay within an acceptable range.
```
def reward_fn(forward_fn,
              params,
              rng,
              data,
              targets,
              total_weight):
    #...
    return reward / total_weight

# And when we call reward_fn we add the new argument:
reward = reward_fn(fwd_fn_apply, test_params, rng,
                           obs, targets, jnp.sum(jnp.absolute(noised_params)))
```
<img src="https://i.imgur.com/rXVxOsV.png">

We can see that the network's total weight is not changing much, now, and since the highest and lowest parameters aren't too big, we can be pretty sure that our updates are still having an impact. How you want to implement weight decay depends a lot on what your network's initial parameters are and what your reward function looks like, and is something that can be toyed with if you feel like your training goes off the rails.

## Use Adam optimization
Adam optimizers (along with some optimizers combining Adam with other techniques) have been responsible for many SOTA models in recent years. Using an Adam optimizer in JAX can be tricky, though. Most implementations of Adam keep some state variables on an object. JAX doesn't let us do objects, so we have to handle that state ourselves. Passing it around through our scan's carry value required a few additional code updates (which I won't paste here, but they can be seen in the repo.) The main logic of the Adam optimizer, though, looks like this:

```
# Based on OpenAI's implementation
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py
def adam_update(grads, state, stepsize=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08):
    (m, v, t) = state
    a = stepsize * jnp.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * (grads * grads)
    step = -a * m / (jnp.sqrt(v) + epsilon)
    return step, (m, v, t+1)
```
For more details on what it's doing, you can check the original paper about it here: https://arxiv.org/abs/1412.6980

We add it to our training routine like so:
```
def run_epoch(carry, rng):
# ...
scaled_noise = get_scaled_noise(reward_out, noise_out)
scaled_noise, adam_state = adam_update(-scaled_noise + .005 * flat_params, adam_state)
return (flat_params + scaled_noise, adam_state), None
```

<img src="https://i.imgur.com/p0M0sOj.png">

With the Adam optimizer, we see better average error at the end, but we see that the network's parameters also managed to grow significantly despite our weight decay scaling. Its adaptive learning rate probably allowed it to "overcome" the weight decay with better overall performance. 

That's more than enough optimization, for this problem. Let's take a look at the main draw of these evolution strategies: running on many GPUs, TPUs, or CPUs in parallel.

# parallel.py

[Click here](https://github.com/ttt733/nn-evolution-jax/blob/main/parallel.py) to view the code.

---

One of the benefits of JAX is that parallelizing functions written for one device can be pretty easy. Right now, vmap is parallelizing our code on a single GPU. We'd like to be able to scale up, though, and for that we'll need its analagous method, [pmap](https://jax.readthedocs.io/en/latest/jax.html?#jax.pmap). We can get how many devices we're working with by using the following code:
```
ndevices = jax.local_device_count()
assert npop % ndevices == 0
```
To make things easy on ourselves, we make sure that our population can be easily divided among our devices. Now, it's just a matter of dividing our population's RNG values up and wrapping our training function with pmap to send them out to different devices. (I like to use einops for this, but if you're more of a fan of classic numpy reshapes, those work too.)
```
r = jax.vmap(run_member, in_axes=(
    0, None, None, None), out_axes=(0, 0))
p = jax.pmap(r, in_axes=(0, None, None, None), out_axes=(0, 0))
population_rng = einops.rearrange(population_rng, '(a b) c -> a b c', a=ndevices)
noise_out, reward_out = p(population_rng, flat_params, obs, targets)
noise_out = einops.rearrange(noise_out, 'a b c -> (a b) c')
reward_out = einops.rearrange(reward_out, 'a b c -> (a b) c')
```
That's all there is to parallelization, at its most basic level. However, we'd like to eventually train networks with more than a few hundred parameters, and communicating those between distributed devices could get slow. Another thing we have to consider is that, in our get_scaled_noise function, we're asking a single device to do some calculations on all of the parameters from our entire population at once. OpenAI talked about training with thousands of devices, and at that kind of scale, that jnp.dot is going to get expensive fast. To handle these issues, we're going to change the code so that it only ever passes RNG values - not the parameter noise they correspond to - between devices, a technique that we could also use to run this in a distributed cluster, and we'll batch out the get_scaled_noise operations.

### RNG Passing
Our goal is to never send our huge parameter arrays between devices. JAX's RNG makes this convenient - if we give it the same RNG values, it'll give us the same output every time. That means we can just pass the RNG values back, and then on the device compiling the results of our population, it can reconstruct the noise of the individual members. First off, we'll change run_member so that it returns the RNG value rather than the noise:
```
return rng, jnp.asarray([reward, neg_reward])
```
And now, with a minor change to get_scaled_noise, we'll have it reconstruct the noise before its calculation. This will slow things down a tiny bit if you're on just one device, but the upside of not having to pass the noise across the network will outweigh that cost at distributed scales.
```
def get_scaled_noise(reward_out, rng_out):
    '''
    Given the RNG values for an entire population, returns an array of noise to be applied
    to base_params before the next training epoch.
    '''
    n = jax.vmap(lambda rng: jax.random.normal(
        rng, (total_params,)) * sigma, in_axes=(0,))
    noise_out = n(rng_out)
    ranks = compute_centered_ranks(reward_out)
    weights = jnp.squeeze(ranks[:, 0] - ranks[:, 1])
    return jnp.dot(weights, noise_out)
```

### Batched noise calculation
Now that we're ready to scale up to thousands of workers - or as many as we can afford, at least - we need to make sure we won't run out of memory when we try to pull them back together on a single device. What we'll do first is batch out our get_scaled_noise call and simply average the results before updating our network parameters. We'll add a new flag, first.
```
nnoisebatches = FLAGS.noise_population_batches
assert npop % nnoisebatches == 0
```
And now, we'll reshape our RNG and reward values to account for that.
```
# Batch results before noise calculation
rng_out = einops.rearrange(rng_out, '(a b) c -> a b c', a=nnoisebatches)
reward_out = einops.rearrange(reward_out, '(a b) c -> a b c', a=nnoisebatches)
```
Finally, all we have to do is vmap our call to get_scaled_noise and average the results.
```
s = jax.vmap(get_scaled_noise, in_axes=(0, 0))
scaled_noise = s(reward_out, rng_out)
scaled_noise = jnp.average(scaled_noise, axis=0)
```
You want to be sure your batches are big enough that you're still getting a nice representative sample, as this method isn't quite as precise as calculating ranks for the whole population at once. If you have so many parameters that you can't fit enough copies of noise into memory, the process can be further batched out by splitting the whole noise array. We'll add another flag:
```
nnoisesplits = FLAGS.noise_split_batches
```
Since we usually don't want to consider exactly how many parameters we'll have, we'll have to pad them out a bit to ensure they divide evenly. It's a little ugly, but it's a small price to pay for being able to train at such a huge scale. We can do it all in our get_scaled_noise function:
```
def get_scaled_noise(reward_out, rng_out):
    '''
    Given the RNG values for an entire population, returns an array of noise to be applied
    to base_params before the next training epoch.
    '''
    def reconstruct_noise(rng):
        noise = jax.random.normal(rng, (total_params,)) * sigma
        # Pad the noise so we can split it evenly
        noise = jnp.concatenate((noise, jnp.zeros(noiseremainder)))
        noise = einops.rearrange(noise, '(a b) -> a b', a=nnoisesplits)
        return noise

    n = jax.vmap(reconstruct_noise, in_axes=(0,))
    noise_out = n(rng_out)
    ranks = compute_centered_ranks(reward_out)
    weights = jnp.squeeze(ranks[:, 0] - ranks[:, 1])
    w = jax.vmap(lambda noise, weights: jnp.dot(
        weights, noise), in_axes=(1, None))
    scaled_noise = w(noise_out, weights)
    # Put our noise back together and slice off the padding
    scaled_noise = einops.rearrange(
        scaled_noise, 'a b -> (a b)', a=nnoisesplits)[:-noiseremainder]
    return scaled_noise
```
Now, we can chop our update calculations up as much as we need to in order to fit it all on a device. The optimal settings for this will be application-specific - it might not even be a bad idea to pmap some of it, if the data is split enough - but they can be calculated after you know how many parameters your model has and what population sizes allow it to learn well.

### Results
Let's do a quick sanity check and make sure we didn't ruin our results.

<img src="https://i.imgur.com/VHN5pPJ.png">

Performance is a little worse, which is to be expected when batching updates, and it's still within an acceptable range for this problem. (Given the jittering we do on the input data, some error is to be expected.) But what's more interesting is the fact that the network's total weight shrank significantly. Batching out our update calculations seems to allow the weight decay to really impact the network - so much so that we might get worried that it's getting too small.

There are some things we could try to fix it, but weight decay calculations are something that will need to be tailored to any target application. It's part of the reward function, after all. We can see in the output that none of the parameters have grown too out of control, so in this case, we probably don't need to worry about it. In a more complex application, you'd want to look at it with a little more scrutiny, to make sure it isn't cutting the time during which your network can learn efficiently short.

---

# The end!
JAX is a powerful tool, and I hope that it can help people find more value in evolution strategies for network training. The ability of evolution strategies to tackle just about any problem, differentiable or not, using the same very basic techniques described here is something I find really interesting. Perhaps the ease with which we can parallelize it across devices is something that will let people who aren't quite masters of reinforcement learning do some cool things with it.
