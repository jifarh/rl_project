"""Adding Explainability"""

import torch
import shap
import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

# Set the environment as LunarLander-v2
env = make_vec_env("LunarLander-v2", n_envs=1)

# Feature names for LunarLander-v2 (source: LunarLander GitHub repository)
feature_names = ['Pos X', 'Pos Y', 'Vel X', 'Vel Y', 'Angle', 'Angular Vel', 'Left Leg Contact', 'Right Leg Contact']

# Generate a sample of observations for SHAP
# observations_sample are the reference samples/background samples
observations_sample = np.array([env.observation_space.sample() for _ in range(500)])

# Choose a new observation to explain
number_of_samples = 500
observation_to_explain = np.array([env.observation_space.sample() for _ in range(number_of_samples)])

###DQN

# Define the model
model = DQN.load("log_dir_DQN/best_model.zip", env=env)

# The value of the best possible action according to the model's current policy is the output of the function
# This is the maximum Q-value across all possible actions.
# We are interested in calculating the importance of each feature toward the maximum Q-value
def model_predict_DQN(observation):
    # Ensure the policy network is in evaluation mode
    model.policy.eval()

    observation = observation.reshape(-1, 8)
    with torch.no_grad():
        q_values = model.policy.q_net(torch.tensor(observation).to(model.device)).cpu().numpy()
        # Take the maximum Q-value across the actions for each observation
        max_q_values = np.max(q_values, axis=1)
    return max_q_values

# Initialize SHAP explainer with the KernelExplainer
explainer = shap.KernelExplainer(model_predict_DQN, observations_sample)

# Global Explainability
# Calculate SHAP values for the entire dataset (reference)
shap_values = explainer.shap_values(observations_sample)

# Create global plots
shap.summary_plot(shap_values, features=observations_sample, plot_type="dot", feature_names=feature_names)
shap.summary_plot(shap_values, features=observations_sample, plot_type="bar", feature_names=feature_names)

# local Explainability
shap_values_local = explainer.shap_values(observation_to_explain)

# Create local plots
# A random index of the observation (from 0 to number_of_samples-1)
obs_index = 22

# Generate a waterfall plot
shap.plots.waterfall(shap.Explanation(values=shap_values_local[obs_index],
                                      base_values=explainer.expected_value,
                                      data=observation_to_explain[obs_index],
                                      feature_names=feature_names))



###A2C
# In actor-critic : an actor decides which action to take, and a critic evaluates the taken action.
# In this function, the value of the state is estimated by the critic as the output.
# This state value is a prediction of the expected returns from that state, following the current policy.
# We are interested in calculating the importance of each feature toward the estimated state value.
def model_predict_A2C_PPO(states):
    # Ensure the input states are in the right shape and type
    states_tensor = torch.as_tensor(states).float().to(model.device)
    with torch.no_grad():
        # Call the forward method of the policy to get the state value estimate
        _, state_value_estimate, _ = model.policy.forward(states_tensor)
        # Convert the state value estimate to a numpy array
        state_value_estimate = state_value_estimate.cpu().numpy()
    return state_value_estimate

# Define the model
model = A2C.load("log_dir_A2C/best_model.zip", env=env)

# Initialize a SHAP explainer
explainer_A2C_PPO = shap.KernelExplainer(model_predict_A2C_PPO, observations_sample)

# Calculate SHAP values for the entire dataset
shap_values = explainer_A2C_PPO.shap_values(observations_sample)
# Extract the SHAP values from the list
shap_values_array = shap_values[0]

# Create global plots
shap.summary_plot(shap_values_array, features=observations_sample, plot_type="dot", feature_names=feature_names)
shap.summary_plot(shap_values_array, features=observations_sample, plot_type="bar", feature_names=feature_names)

# Create local plots
# Calculate SHAP values for this observation
shap_values_local = explainer_A2C_PPO.shap_values(observation_to_explain)
shap_values_local_array = shap_values_local[0]

# Visualization
# Same index of the observation
obs_index = 22
# Generate a waterfall plot for the observation's SHAP values
shap.plots.waterfall(shap.Explanation(values=shap_values_local_array[obs_index],
                                      base_values=explainer_A2C_PPO.expected_value,
                                      data=observation_to_explain[obs_index],
                                      feature_names=feature_names))


### PPO

#Define the model
model = PPO.load("log_dir_PPO/best_model.zip", env=env)

# Initialize SHAP explainer
explainer_A2C_PPO = shap.KernelExplainer(model_predict_A2C_PPO, observations_sample)

# Calculate SHAP values for the entire dataset
shap_values = explainer_A2C_PPO.shap_values(observations_sample)
# Extract the SHAP values from the list
shap_values_array = shap_values[0]

# Create Global plots
shap.summary_plot(shap_values_array, features=observations_sample, plot_type="dot", feature_names=feature_names)
shap.summary_plot(shap_values_array, features=observations_sample, plot_type="bar", feature_names=feature_names)

# Create local plots

# Calculate SHAP values for this observation
shap_values_local = explainer_A2C_PPO.shap_values(observation_to_explain)
shap_values_local_array = shap_values_local[0]

# Visualization
# Same index of the observation
obs_index = 22
# Generate a waterfall plot for this observation
shap.plots.waterfall(shap.Explanation(values=shap_values_local_array[obs_index],
                                      base_values=explainer_A2C_PPO.expected_value,
                                      data=observation_to_explain[obs_index],
                                      feature_names=feature_names))