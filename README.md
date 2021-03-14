# Novel-Reinforcement-Learning-Algorithm-on-Google-Football-Research

## Introduction
In the study of reinforcement learning, the main goal is to train smart agents that can solve complex tasks in the process of interacting with the environment. While the real-world environment is still complicated for common reinforcement learning algorithms, great progress has been made for various games, such as the iconic Atari console games, the ancient game of Go, professional played games Starcraft II and Dota 2. These games provide challenging and reproducible environments for us to test new ideas and novel algorithms.

While all the reinforcement learning environments mentioned above have been studied comprehensively, our project is based on Google Research Foot- ball Environment, a novel reinforcement learning environment created by AI team from Google Research in 2019. In this environment, agents aim to master one of the world’s most popular sports, football. Modeled after pop- ular football video games, the Football Environment provides an advanced, physics-based 3D football simulation where agents control either one or all football players on their team, learn how to pass between them, and manage to overcome their opponent’s defense in order to score goals.

![figure1](https://github.com/WallaceSUI/Novel-Reinforcement-Learning-Algorithm-on-Google-Football-Research/blob/main/figures/figure1.png)

## Our Methods
Based on the limitation of computer resources, we choose DQN as our start point. In this section, we will introduce the whole pipeline of our model and three improvements based on the model design.

### Data Preprocessing and Feature Engineering
In the original input data, there are four images in each frame representing our observations, as is shown in Fig. 2. In figure ”left team” and ”right team”, there are 11 white pixels representing the current locations of each player in our team and the opponent’s team. In figure ”ball”, one white pixel represents the location and movement of the ball. In figure ”active player”, the player that we control can be seen as a white pixel. Based on these observations, different engineering techniques can be considered.

First, our inputs data are composed of many continuous frames and each frame is composed of 4 images as observations. For a video-based dataset, it will have temporal-based features and time-based features. For the temporal-based features, our model can know the location of players and ball in the images. we also extract the motion-related features to give the motion information to the model. This can let the model know the direction of movement of each player in the image. For the time-based features, we extract a batch of continuous frames as one single input to let the model understand what happens in 20-30 continuous frames over times. Second, in each frame, active player doesn’t know its relative position to the opponent’s player. Based on this finding, we add features to establish a connection between active player’s location and opponent players’ locations. Third, we give some penalty labels such as offside Boolean flag for all friendly players. This means that a player will be penalized with a free kick upon controlling the ball due to the offside rule. Based on these conditions, we aggregate different information into our inputs data to let our model have a better understanding about our game (Figure 3).

![figure2](https://github.com/WallaceSUI/Novel-Reinforcement-Learning-Algorithm-on-Google-Football-Research/blob/main/figures/figure2.png)

### Model Design and Training
To improve the feature extraction ability, we used a convolutional neural network based structure to design our whole model. Details are in Figure 4. For each input graph, we first use a layer to separate inputs into 4 different branches and then we use a multi-layer convolutional network to extract features. The first branch is ”teammate block”, which has 3 convolutional layers to extract features about the teammates’ location and their move- ment. The second branch is ”opponents block”, which has 3 convolutional layers over 11 opponents using their positions and velocities relative to the active player as features. The third branch shows the ball location feature extraction and the fourth branch is the active player block, which has the features from the active player’s point of view and direction of movement. The kernel size of convolutional layers are 8, 4, 3 respectively and we set our batch size to be 75. After the different branches’ feature extraction, fully-connected layers are trained to combine all features extracted from the previous ones and output final action prediction. During the training, we set different number of episodes and max steps to fine-tune our model and we use the MSE loss between predicted Q value and optimal Q value from Bellman equation to update our model. For each single model, we use our own single GPU for training and it needs nearly 2 days for one experiment since it is very slow during each updating.

![figure3](https://github.com/WallaceSUI/Novel-Reinforcement-Learning-Algorithm-on-Google-Football-Research/blob/main/figures/figure3.png)

![figure4](https://github.com/WallaceSUI/Novel-Reinforcement-Learning-Algorithm-on-Google-Football-Research/blob/main/figures/figure4.png)

### Replay Buffer Redesign
During the training, we should use replay buffer to store our sampling. One of biggest problems in replay buffer is the unbalanced sampling. If the process of sampling is not controlled, most of data are zero rewarded. Based on this condition, it is hard for our model to learn the positive and negative way of the game and thus it will reduce our final performance. Therefore, we redesign our replay buffer by changing the ratio of positive samples and negative samples. In the new buffer, we have three different first-in-first-out queues to store the different part of data and sample them with replacement to balance the proportion.

### Rewards Redesign
Besides modification on the buffer sample strategy, we also tried to introduce the physical meanings of each actions. Although the split in positive and negative samples enabled us to adjust the proportion of positive rewards and granted the DQN tendency to be active, there were two issues underlying the modification of sample strategy: 1. Limited actions like shooting dom- inated the entire action space, which caused the controlled team had less time to dribble into the opposite court. 2. The distribution of actions was monotonous, which were almost determined by the design of sample strat- egy and reduced the effect of learning. To solve this problems, we found modification on reward enabled us to regularize the action distribution with more flexibility.

First, we encouraged the controlled player to run towards the ball. Ini- tially we only awarded the directional actions running towards the ball, but we found that the player sprinted over and often chased the ball, which might be caused by that the oscillation among the ball would results in a higher rewards than reach the ball in minimum time. Then we decided to award the actions based on the distance between the ball and the controlled player. The rewards were continuously added for each step and linear pro- portional to the distance between ball and player. Similarly, when the ball gets closer to the opponent’s goal, it is more likely for our team to score. Hence we added rewards based the distance between the ball and opponent’s goal. These two rewards should be adjustable and settled in every step.

Then, we analyzed the play video of agent and discovered that although the ball is successfully carried close to the opponent’s goal, the controlled player often hovers in the forbidden area without shooting. We regarded this as the modified distance reward dominated the positive rewards and shooting might result in the reduction of possession and increase the distance between the ball and opponent’s goal. Hence once the controlled player shooting in the forbidden area, we add rewards. Meanwhile, idle actions make would not benefit the training process of agent, so we heavily penalized the rewards when idle action occurs.

### Rule Based Initialization
After redesigning the reward, we still cannot get accurate q values due to the computing resources limitation, which can be observed in Figure 7 with very low proportions of actions ’short pass’, ’high pass’ and ’right’ after 30 epochs. Therefore, to speed up the training process, and push the model to learn how to defend, pass and shot within limit training, we replace the random initialization in epsilon greedy part with probability 0.5 by a easy rule based initialization to help the model get better training data at the beginning.

## Experiments
### After Feature Engineering and Model Design
Based on our engineering techniques and convolutional network design, we visualize the actions prediction during each step (Figure 5). From the visu- alization, we can see that many predicted actions are ”idle” or ”run left”, which means that our players do more defending than scoring. This is be- cause that every time when our model predicts the action ”shot”, it is hard for our team to get the scoring so that the loss will not encourage our model to do more shooting in the next step. Although our feature engineering and model design can help our model understand what happens during the game, we still need other improvements to let the model try to find the chance to get the scoring.

![figure5](https://github.com/WallaceSUI/Novel-Reinforcement-Learning-Algorithm-on-Google-Football-Research/blob/main/figures/figure5.png)

### After Replay Buffer Redesign
After changing the distribution of different data ratio in our replay buffer, we can see that the model has a different actions prediction distribution (Figure 6). From this result, we can see that our model can predict action ”shot” in a higher probability and the bad actions like ”idle” and ”left” reduce a lot after we change the design in our replay buffer.

![figure6](https://github.com/WallaceSUI/Novel-Reinforcement-Learning-Algorithm-on-Google-Football-Research/blob/main/figures/figure6.png)

### After Rewards Redesign
Comparing the plots, we successfully controlled the distribution of ac- tion frequency as shown. The modification on rewards not only increase or decrease the frequency of actions with modified rewards, but also enlighten the agent find some useful techniques like sliding and long pass.

![figure7](https://github.com/WallaceSUI/Novel-Reinforcement-Learning-Algorithm-on-Google-Football-Research/blob/main/figures/figure7.png)

### After Rule Based Initialization
Figure 8 shows the frequency of each action after 30 epochs. From the figure we can see that there are significant increases in actions ’right’, ’short pass’, ’high pass’, and a significant decrease in action ’release sprint’, which shows our initialization pushes the model to learn how to attack better in the beginning.

![figure8](https://github.com/WallaceSUI/Novel-Reinforcement-Learning-Algorithm-on-Google-Football-Research/blob/main/figures/figure8.png)

## Results
![figure9](https://github.com/WallaceSUI/Novel-Reinforcement-Learning-Algorithm-on-Google-Football-Research/blob/main/figures/figure9.png)
