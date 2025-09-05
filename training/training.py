import os
import ray
import time
from ray.rllib.algorithms.ppo import PPOConfig
from sim_env.parking_env import Parking
from sim_env.parameters import Config, PI
from utility import custom_log_creator, custom_log_checkpoint, create_folder_path, create_folder_name


ray.init()
env_name = Parking
config = Config(car_length=4.0, car_width=2.0,
                wheel_length=0.75, wheel_width=0.35,
                parking_length=6.0, parking_width=2.2,
                max_distance=25.0, max_steps=900,
                acceleration_limit=1.0, steering_limit=0.59, velocity_limit=0.6,
                max_angle_error=PI/12, center_threshold=0.2, penalty_ratio={'angle': 0.35, 'velocity': 0.15, 'segment': 0.2, 'steps': 0.01, 'steering': 0.00, 'acceleration': 0.00},
                reward_type='type4', state_type='type4',
                side=1, car_loc_randomize_range=(10.0, 10.0), initial_distance_range=(2.5, 2.5)
                )
env_config = {"render_mode": "no_render",
              "action_type": "continuous",
              "parking_type": "parallel",
              "training_mode": "on",
              'config': config}

# for folder names
num_train = 200
side = config.side
threshold = config.center_threshold
angle_ratio, v_ratio, seg_ratio, steps_ratio, steering_ratio, acceleration_ratio = config.penalty_ratio['angle'], config.penalty_ratio['velocity'], config.penalty_ratio['segment'], config.penalty_ratio['steps'], config.penalty_ratio['steering'],config.penalty_ratio['acceleration']

folder_path = create_folder_path(env_config, is_training=True)
folder_name = create_folder_name('PPO', env_config, config.reward_type, config.state_type,
                                 num_train, side, folder_path, threshold, angle_ratio, v_ratio, seg_ratio, steps_ratio, steering_ratio, acceleration_ratio)

algo = (
    PPOConfig()
    .environment(env=env_name, env_config=env_config)
    .rollouts(num_rollout_workers=1)
    # .training(
    #     lr=1e-4,
    #     gamma=0.99,
    #     lambda_=0.95,
    #     clip_param=0.2,
    #     vf_loss_coeff=0.5,
    #     entropy_coeff=0.001,
    #     train_batch_size=4000,
    #     sgd_minibatch_size=128,
    #     num_sgd_iter=15,
    # )
    .training(
        lr=5e-5,
        gamma=0.99,
        lambda_=1.0, #0.95,
        clip_param=0.2, #0.2,
        vf_loss_coeff=1.0, #0.5,
        entropy_coeff=0.0, #0.001,
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=30, #15,
    )
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .framework("torch")
    .evaluation(evaluation_num_workers=1)
    .build(logger_creator=custom_log_creator(folder_path, folder_name))
)

start_time = time.time()
# training
for i in range(int(num_train)):
    print("Iterations:", i, ":", algo.train())
end_time = time.time()
print(f"Total execution time of the script: {end_time - start_time} second")

algo.evaluate()

# save the checkpoint
folder_path = folder_path.replace('/training_results', '/trained_agents')
checkpoint_dir = custom_log_checkpoint(folder_path, folder_name)
checkpoint_dir = algo.save(checkpoint_dir)
print(f"Checkpoint saved in directory {checkpoint_dir}")
