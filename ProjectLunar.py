# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import shap
import numpy as np
# import gym

# Step 1: Initialize the environment
env = make_vec_env('LunarLander-v2', n_envs=1)

# Step 2: Initialize the model
model = DQN("MlpPolicy", env, verbose=1)

# Step 3: Train the model
model.learn(total_timesteps=10000)

# Step 4: Save the model (optional)
model.save("dqn_lunarlander")

# Evaluating the Agent
# Load the trained model
model = DQN.load("dqn_lunarlander")

# Evaluate the agent
episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        # env.render()  # Uncomment this if you want to see the game window
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        score += rewards
    print(f'Episode:{episode} Score:{score}')

# Function to wrap your model's Q-value predictions
def model_predict(observation):
    # The LunarLander observation space has a shape of (8,)
    observation = observation.reshape(-1, 8)  # Reshapes to (1, 8) if a single observation is passed
    actions, _states = model.predict(observation, deterministic=True)
    return actions

# Feature names for LunarLander-v2
feature_names = ['Pos X', 'Pos Y', 'Vel X', 'Vel Y', 'Angle', 'Angular Vel', 'Left Leg Contact', 'Right Leg Contact']

# Generate a sample of observations for SHAP
observations_sample = np.array([env.observation_space.sample() for _ in range(100)])

# Initialize SHAP explainer with the KernelExplainer or another appropriate explainer
explainer = shap.KernelExplainer(model_predict, observations_sample)

# Choose an observation to explain
observation_to_explain = env.reset()

# Calculate SHAP values for this observation
shap_values = explainer.shap_values(observation_to_explain)

# Visualization
# Save the force plot as an HTML file
shap_plot = shap.force_plot(explainer.expected_value, shap_values, observation_to_explain, show=False)
shap.save_html('shap_force_plot.html', shap_plot)

# Generate a sample of observations for SHAP (repeated as it seems intended to be used later)
observations_sample = np.array([env.observation_space.sample() for _ in range(100)])

# Initialize SHAP explainer again (may not be necessary to repeat this step)
explainer = shap.KernelExplainer(model_predict, observations_sample)

# Choose an observation to explain (this seems redundant and can be removed)
observation_to_explain = env.reset()

# Calculate SHAP values for this observation again (this is also redundant and can be removed)
shap_values = explainer.shap_values(observation_to_explain)

# Use SHAP's summary plot to display feature impact with names
shap.summary_plot(shap_values, features=observation_to_explain.reshape(1, -1), feature_names=feature_names)

# After computing shap_values and choosing an observation_to_explain...
# Summary plot
shap.summary_plot(shap_values, features=observation_to_explain, feature_names=feature_names)

# Decision plot
shap.decision_plot(explainer.expected_value, shap_values, features=observation_to_explain, feature_names=feature_names)

# Dependence plot - Plotting for the first feature as an example
shap.dependence_plot(0, shap_values, features=observation_to_explain, feature_names=feature_names)

# Other SHAP plots can be generated in a similar manner as needed.

