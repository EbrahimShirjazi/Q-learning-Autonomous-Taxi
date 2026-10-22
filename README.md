# Q-Learning Autonomous Taxi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EbrahimShirjazi/Q-learning-Autonomous-Taxi/blob/main/q_learning_taxi_v3.ipynb)

A tabular Q-learning agent that learns to pick up and drop off passengers in Gymnasium's [Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/). Built while following Unit 2 of the [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/en/unit2/hands-on).

**Result:** mean reward **8.20 ± 2.39** over 200 test episodes from random starts.

Trained agent on Hugging Face: [Ebishj/taxi-v3advanced-Q-learning](https://huggingface.co/Ebishj/taxi-v3advanced-Q-learning)

## Replay



https://github.com/user-attachments/assets/60ff1661-ec23-4df8-ab6a-c575fab1c47a



## Learning curve

<img width="707" height="393" alt="download (1)" src="https://github.com/user-attachments/assets/a37eb42c-ce47-469f-a6cd-efcbe03a237e" />


## Files

- `q_learning_taxi_v3.ipynb`: the full notebook (training, evaluation, learning curve, replay)
- `qtable_taxi.npy`: the trained Q-table (500 states × 6 actions)
- `replay.mp4`: one episode played by the trained agent

## Run it

Open the notebook in Colab with the badge above and choose **Runtime → Run all**. It installs what it needs, including `gymnasium==1.2.3`, the last version with Taxi-v3.

## Use the trained Q-table

```python
import numpy as np
import gymnasium as gym  # gymnasium==1.2.3

Qtable = np.load("qtable_taxi.npy")
env = gym.make("Taxi-v3")
state, info = env.reset()
action = np.argmax(Qtable[state])  # the best action in this state
```
