# Evolving Neural Networks in JAX
This repository holds code related to an article describing techniques for applying evolutionary network training strategies in JAX. Each script trains a network the same problem - given a sequence of regularly-spaced values on a sine wave, predict the next value. Much of the code is duplicated between scripts so that, if they like, readers can view the differences between files to see what changes in each section. The algorithms themselves are taken from [OpenAI's blog post](https://openai.com/blog/evolution-strategies/) describing their efforts at scaling evolution strategies. 

## simple.py
In this file, a very basic evolutionary strategy is implemented, without many optimizations. You can get a grasp here on how some fundamental JAX methods like `scan` and `vmap` are used to execute our training routine.

## advanced.py
Here, some optimizations that OpenAI made in their code are added to our training routine. The various optimizations are discussed in depth in the article.

## parallel.py
In this file, we prepare to scale the network to more than one device and to greater sizes. Vectorization becomes parallelization, and the code is sliced up so that we can calculate our network updates on a single device.