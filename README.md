# Code for "ROBUST Q-LEARNING FOR FINITE AMBIGUITY SETS"

## Cécile Decker, Julian Sester

# Abstract

In this paper we propose a novel $Q$-learning algorithm allowing to solve distributionally robust Markov decision problems for which the ambiguity set of probability measures can be chosen arbitrarily as long as it comprises only a finite amount of measures. Therefore, our approach goes beyond the well-studied cases involving ambiguity sets of balls around some reference measure with the distance to reference measure being measured with respect to the Wasserstein distance or the Kullback--Leibler divergence. Hence, our approach allows  the applicant to create ambiguity sets better tailored to her needs and to solve the associated robust Markov decision problem via a $Q$-learning algorithm whose convergence is guaranteed by our main result. Moreover, we showcase in several numerical experiments the tractability of our approach.

# Preprint

[Link](https://arxiv.org/abs/2407.04259)

# Content

The Examples from the paper are provided as seperate jupyter notebooks, each with a unique name, exactly specifying which example is covered therein. These are:
- An [Example 4.1](https://github.com/CecileDecker/FiniteQLearning/blob/main/Example_4.1_cointoss.ipynb) covering finite Q learning for a coin toss game (Example 4.1 from the paper).
- An [Example 4.2](https://github.com/CecileDecker/FiniteQLearning/blob/main/Example_4.2_stockinvesting.ipynb) covering finite Q learning for a stock investing example (Example 4.2 from the paper).
