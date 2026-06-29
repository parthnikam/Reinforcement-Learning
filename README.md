# Week 4 - Learning RL
## Parth Nikam 

Github: `https://github.com/parthnikam/Reinforcement-Learning`


## Overview: 
This week I got into reading the book `Deep Reinforcement Learning` and understood some of the fundamental concepts from it.<br/>
Also watched a couple of lectures on youtube on RL to understand policy based and value based learning functions.<br/>
Trained agents on `gymnasium` library's lunar lander and racecar example. 


## Concepts I learnt: 
1. Learnt about Markov Decision Process and Monte Carlo Optimization
2. Learnt Q Learning equation and how it's used to train an agent loop
3. How to set up rewards and actions based on the state the agent is acting in 
4. How batch training optimizes training for agent over multiple epochs - learnt weights are carried over to the next epoch and stored to prevent catastrophic forgetting 
5. Modern RL just uses QLearning to find the right actions but they have to be learnt by a neural network in the end - Neural Networks are the brains of the agent!
6. Making multiple agents run concurrently speeds up exploration and their weights can be shared


## What I built:
1. built a terminal gridworld in python and implemented a basic Q Learning policy that learnt gridworld in 100 episodes 
2. tried implementing policy based learning algorithm on lunar lander - it was pretty straight forward 
  - implemented player, agent and training loop - to play the game yourself, watch the agent play and train the agent 
  - save the learnt weights and load it for next session 
3. took gymnasium's racecar game and implemented actor critic PPO on it. 
  - it learnt how to drive around the track pretty decently in 100 training episodes 
  - used a CNN neural network as the brain of the agent 
  - trained it in batches of 64
4. implemented a multi-agent learning system where multiple agents would learn in the same environment and share weights to speed up training process 


## Challenges I faced: 
1. understanding how the hyperparameters affect the agent's learning was pretty straightforward but actually having the right set of hyperparameters during training was daunting. and I ended up outsourcing those decisions to LLM. 
2. implementing PPO and reward functions was not straightforward even for a simple game like racecar 
3. training 100 episodes took quite a while for simple a racecar game - real world RL systems might take an enormous amount of compute and time!
