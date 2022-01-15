# Evolving Neural Networks in JAX
This repository holds code displaying techniques for applying evolutionary network training strategies in JAX. Each script trains a network to solve the same problem: given a sequence of regularly-spaced values on a sine wave, predict the next value. The problem is trivial - the interesting part is intended to be the way in which this is accomplished, by updating network parameters directly and without gradient calculations, in parallel across devices. A lengthy tutorial is included, explaining the ideas and rationale. Much of the code is duplicated between scripts so that readers can run them individually and, if they like, view the differences between files to see what changes in each section.

The evolutionary ideas present here are mainly taken from [OpenAI's blog post](https://openai.com/blog/evolution-strategies/) describing their efforts at scaling evolution strategies (and the associated code.)

## tutorial.md
A longform tutorial that explains why I think evolutionary optimization strategies are interesting and some of the JAX techniques that I use to implement them. Individual bits of the code in each of the script files are discussed here.

## simple.py
In this file, a very basic evolutionary strategy is implemented, without many optimizations. You can get a grasp here on how some fundamental JAX methods like `scan` and `vmap` are used to execute our training routine.

## advanced.py
Here, some optimizations that OpenAI made in their code are added to our training routine. The various optimizations are discussed in depth in the article.

## parallel.py
In this file, we prepare to scale the network to more than one device and to greater sizes. Vectorization becomes parallelization, and the code is sliced up so that we can calculate our network updates on a single device.