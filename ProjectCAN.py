# import json
#
# def load_json_data(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data
#
# # Example of how to use the function to load one file
# pose_data = load_json_data(r'C:\Users\Aurora\Desktop\RL\Dataset\can_bus\can_bus\scene-1109_pose.json')
#
# print(pose_data)


# Step 1: Parsing JSON Files and Feature Extraction
import json
import numpy as np

# Paths to the uploaded files
file_paths = {
    'meta': r'C:\Users\Aurora\Desktop\RL\Dataset\can_bus\can_bus\scene-1110_meta.json',
    'ms_imu': r'C:\Users\Aurora\Desktop\RL\Dataset\can_bus\can_bus\scene-1110_ms_imu.json',
    'pose':r'C:\Users\Aurora\Desktop\RL\Dataset\can_bus\can_bus\scene-1110_pose.json',
    'route': r'C:\Users\Aurora\Desktop\RL\Dataset\can_bus\can_bus\scene-1110_route.json',
    'steeranglefeedback': r'C:\Users\Aurora\Desktop\RL\Dataset\can_bus\can_bus\scene-1110_steeranglefeedback.json',
    'vehicle_monitor': r'C:\Users\Aurora\Desktop\RL\Dataset\can_bus\can_bus\scene-1110_vehicle_monitor.json',
    'zoe_veh_info': r'C:\Users\Aurora\Desktop\RL\Dataset\can_bus\can_bus\scene-1110_zoe_veh_info.json',
    'zoesensors': r'C:\Users\Aurora\Desktop\RL\Dataset\can_bus\can_bus\scene-1110_zoesensors.json'
}

# Function to load data from a given filepath
def load_json_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Load all data
all_data = {name: load_json_data(filepath) for name, filepath in file_paths.items()}
# Print the keys of the all_data dictionary
print("Data keys loaded:", all_data.keys())

# Print the type and length/size of each item in all_data
for key, value in all_data.items():
    print(f"{key}: Type = {type(value)}, Length/Size = {len(value) if hasattr(value, '__len__') else 'N/A'}")


# Extract features (as an example, we are taking a couple of features from vehicle_monitor and ms_imu)
# This structure needs to be defined based on the specific features you want to use
state_features = {
    'vehicle_speed': [d.get('vehicle_speed') for d in all_data['vehicle_monitor']],
    'acceleration': [d.get('accel') for d in all_data['ms_imu']],  # Assuming 'accel' is the correct key
    # Add other features as required
}

# Convert lists to numpy arrays for efficient numerical computation
for feature_name, feature_data in state_features.items():
    state_features[feature_name] = np.array(feature_data)

#
# # Step 2: Define Action and Reward Strategy
# # Assume we have discrete actions corresponding to [do_nothing, accelerate, brake, turn_left, turn_right]
# # Rewards could be based on maintaining a certain speed, smooth steering, etc.
#
# def compute_reward(state, action):
#     # Placeholder for reward computation logic
#     # Example: penalize deviation from desired speed
#     desired_speed = 30  # 30 m/s as an example
#     reward = -abs(state['vehicle_speed'] - desired_speed)
#     return reward
#
# # Step 3: DQN Architecture Setup
# import tensorflow as tf
# from tensorflow.keras import models, layers, optimizers
#
# # DQN Model
# def create_dqn_model(input_shape, action_space):
#     model = models.Sequential([
#         layers.Dense(64, activation='relu', input_shape=input_shape),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(action_space, activation='linear')  # No. of actions
#     ])
#     model.compile(optimizer=optimizers.Adam(lr=0.001), loss='mse')
#     return model
#
# # Define the state space and action space dimensions
# state_space = len(features)  # Assuming features is a dictionary of numpy arrays
# action_space = 5  # Assuming 5 possible actions: [do_nothing, accelerate, brake, turn_left, turn_right]
#
# # Instantiate the model
# dqn_model = create_dqn_model(input_shape=(state_space,), action_space=action_space)
#
#
# # Step 4: Training the DQN on the Episodes
# from collections import deque
# import random
#
# # Hyperparameters
# epsilon = 1.0  # Exploration rate
# epsilon_min = 0.01
# epsilon_decay = 0.995
# batch_size = 64
# episodes = 100  # Number of episodes to train on
# memory = deque(maxlen=2000)
#
#
# # Function to choose an action based on epsilon-greedy policy
# def choose_action(state, model):
#     if np.random.rand() <= epsilon:
#         return random.randrange(action_space)
#     act_values = model.predict(state)
#     return np.argmax(act_values[0])
#
#
# # Function to replay and train from memory buffer
# def replay(model, memory, batch_size):
#     minibatch = random.sample(memory, batch_size)
#     for state, action, reward, next_state, done in minibatch:
#         target = reward
#         if not done:
#             target = (reward + 0.95 * np.amax(model.predict(next_state)[0]))
#         target_f = model.predict(state)
#         target_f[0][action] = target
#         model.fit(state, target_f, epochs=1, verbose=0)
#     global epsilon
#     if epsilon > epsilon_min:
#         epsilon *= epsilon_decay
#
#
# # Training loop
# for e in range(episodes):
#     # Reset the environment for a new episode
#     # Assume we have a function `env_reset` that returns the initial state
#     state = env_reset(features)
#     state = np.reshape(state, [1, state_space])
#
#     for time in range(200):  # Assuming max timesteps per episode is 200
#         action = choose_action(state, dqn_model)
#         next_state, reward, done, _ = env_step(action)  # Implement `env_step` based on your environment logic
#         reward = compute_reward(state, action)
#
#         next_state = np.reshape(next_state, [1, state_space])
#         memory.append((state, action, reward, next_state, done))
#         state = next_state
#
#         if done:
#             print(f"episode: {e}/{episodes}, score: {time}, e: {epsilon:.2}")
#             break
#
#         if len(memory) > batch_size:
#             replay(dqn_model, memory, batch_size)
#
