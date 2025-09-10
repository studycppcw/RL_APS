import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # required on macOS if duplicate libomp is unavoidable


import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import time
from ray.rllib.algorithms.ppo import PPO
from parameters import Config, PI
from parking_env import Parking
from training.utility import create_folder_path

# perpendicular type 4
# config = Config(car_length=4.0, car_width=2.0,
#                 wheel_length=0.75, wheel_width=0.35,
#                 parking_length=6.0, parking_width=4.0,
#                 max_distance=10.0, max_steps=80,
#                 acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=10.0,
#                 max_angle_error=PI/12, center_threshold=0.5, penalty_ratio={'angle': 0.25, 'velocity': 0.25},
#                 reward_type='type4', state_type='type4',
#                 side=1, car_loc_randomize_range=(-5.0, 5.0), initial_distance_range=(5.0, 7.0)
#                 )
# env_config = {"render_mode": "human",
#               "action_type": "continuous",  # or "discrete"
#               "parking_type": "perpendicular",
#               "training_mode": "off",
#               'config': config}
# folder_name = 'PPO_perpendicular_continuous_300_r4_s4_all_th05_ar02_vr02' 

# parallel type 4
# config = Config(car_length=4.0, car_width=2.0,
#                 wheel_length=0.75, wheel_width=0.35,
#                 parking_length=6.0, parking_width=4.0,
#                 max_distance=10.0, max_steps=80,
#                 acceleration_limit=1.0, steering_limit=PI/4, velocity_limit=10.0,
#                 max_angle_error=PI/12, center_threshold=1.0, penalty_ratio={'angle': 0.35, 'velocity': 0.15},
#                 reward_type='type4', state_type='type4',
#                 side=1, car_loc_randomize_range=(6.0, 7.0), initial_distance_range=(5.0, 7.0)
#                 )
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=2.2,
                max_distance=25.0, max_steps=900,
                acceleration_limit=1.0, steering_limit=0.59, velocity_limit=0.6,
                max_angle_error=PI/12, center_threshold=0.1, penalty_ratio={'angle': 0.35, 'velocity': 0.15, 'segment': 0.2, 'steps': 0.01, 'steering': 0.0, 'acceleration': 0.00},
                reward_type='type4', state_type='type4',
                side=1, car_loc_randomize_range=(10.0, 10.0), initial_distance_range=(2.5, 2.5)
                )
env_config = {"render_mode": "human",
              "action_type": "continuous",
              "parking_type": "parallel",
              "training_mode": "off",
              'config': config}
folder_name = 'PPO_parallel_continuous_200_r4_s4_b_th01_ar03_vr01_segr02_stpr00_strr00_acclr00_7' #'PPO_parallel_continuous_200_r4_s4_b_th05_ar03_vr01_segr00_stpr00_strr00_acclr00_1' #'PPO_parallel_continuous_100_r4_s4_b_th08_ar04_vr01_2' # #  # trained_agent folder
# env = Parking({**env_config,"render_mode":"human"})
env = Parking(env_config)
env.reset(seed=seed)

folder_path = create_folder_path(env_config, is_training=False)
folder_path = folder_path.replace('sim_env', 'training')

algo = PPO.from_checkpoint(folder_path + folder_name)
algo.config["seed"] = seed

episode_reward = 0
for i in range(10):
    episode_reward = 0
    terminated = truncated = False
    obs, info = env.reset(seed=seed)
    # env.render(episode_reward)
    actions = []
    while not terminated and not truncated:
        # Algorithm.compute_single_action() is to programmatically compute actions from a trained agent.
        action = algo.compute_single_action(obs, explore=False)
        # action = env.action_space.sample()  # env.action_space.sample() is to sample random actions.
        # action = int(input("Action: "))
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        # env.render(reward)
        # print("obs: ", obs, "reward: ", reward, "info: ", info)
        time.sleep(0.1)
        episode_reward += reward
        # print("Episode reward:", episode_reward)
    print(f'episode {i}: {episode_reward}')
