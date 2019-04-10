# Flappy Bird with Deep Reinforcement Learning
Flappy Bird Game trained on a Double Dueling Deep Q Network with Prioritized Experience Replay implemented using Pytorch.

![Gameplay](https://github.com/adityajn105/flappy-bird-deep-q-learning/blob/master/screenshots/gameplay.gif) 

[See Full 3 minutes video](https://youtu.be/a5vtakBxh6Y)

## Getting Started
Here I will explain how to run the game which runs automatically using saved model, also I will gbreif you about basics of Q Learning, Dueling architecture and Prioritized Experience Replay.

### Prerequisites
You will need Python 3.X.X with some packages which you can install direclty using requirements.txt.
> pip install -r requirements.txt

### Running The Game
Use the following command to run the game where '--model' indicates the location of saved DQN model.
> python3 play_game.py --model checkpoints/flappy_best_model.dat

## Deep Q Learning
Q Learning is off policy learning method in reinforcement learning which is a developement over on-policy Temporal Difference control algorithm. Q-learning tries to estimate a state-action value function for target policy that deterministically selects the action of highest value.

The problem with Tradition Q learning is that it is not suitable for continuous environment (like Flappy Bird) where an agent can be in infinite number of states. So it is not feasible to store all states in a grid which we use in tradition Q learning. So we use Deep Q learning in these environments.

Deep Q learning is based on Deep Neural Network which takes current state in the form of image or say continuous value and approximates Q-values for each action based on that state.

![Deep Q Learning](https://cdn-images-1.medium.com/max/800/1*w5GuxedZ9ivRYqM_MLUxOQ.png)

[Take a look at this article which explains Deep Q Learning](https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8)

### Network Architecture (Dueling Architecture)
Here I have used Dueling architecture to calculate Q values. Q-values correspond to how good it is to be at that state and taking an action at that state Q(s,a). 
So we can decompose Q(s,a) as the sum of:
**V(s)** - the value of being at that state
**A(s)** - the advantage of taking that action at that state (how much better is to take this action versus all other possible actions at that state).

```
** Q(s,a) = V(s) + A(s,a) **
```

![Dueling Architecure](https://cdn-images-1.medium.com/max/1200/1*FkHqwA2eSGixdS-3dvVoMA.png)

### Prioritized Experience Replay
The idea behind PER was that some experiences may be more important than others for our training, but might occur less frequently. Because we sample the batch uniformly (selecting the experiences randomly) these rich experiences that occur rarely have practically no chance to be selected. We want to take in priority experience where there is a big difference between our prediction and the TD target, since it means that we have a lot to learn about it.

```
** pt = |dt| + e **
where,
	pt = priority of the experience
	dt = magnitude of TD error
	e = constant assures that priority do not become 0
```


[Take a look at this article which explains Double Dueling and PER](https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682)

## Authors
* Aditya Jain : [Portfolio](https://adityajn105@github.io)

## Licence
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/adityajn105/flappy-bird-deep-q-learning/blob/master/LICENSE) file for details

## Acknowledements
* The Game has been taken from this [repository](https://github.com/sourabhv/FlapPyBird)
* Thanks Siraj Raval for Move37 course on theschool.ai which helped understand these concepts.