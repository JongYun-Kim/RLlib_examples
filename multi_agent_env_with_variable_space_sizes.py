import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.env import MultiAgentEnv
from gym.spaces import Box
import numpy as np


class VariableSpaceEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.max_observation_space = Box(low=-np.inf, high=np.inf, shape=(1000,))
        self.max_action_space = Box(low=-1.0, high=1.0, shape=(1000,))
        self.observation_space = self.max_observation_space
        self.action_space = self.max_action_space
        self.episode_length = config["episode_length"]
        self.num_agents = config["num_agents"]
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

    def reset(self):
        # Determine the observation and action space sizes for each agent
        obs_sizes = np.random.randint(1, 7, size=self.num_agents)
        action_sizes = np.random.randint(1, 11, size=self.num_agents)

        # Update the observation and action spaces with the correct sizes for each agent
        self.observation_space = {agent: Box(low=-np.inf, high=np.inf, shape=(obs_sizes[i],)) for i, agent in enumerate(self.agents)}
        self.action_space = {agent: Box(low=-1.0, high=1.0, shape=(action_sizes[i],)) for i, agent in enumerate(self.agents)}

        # Return the initial observations for each agent
        return {agent: np.random.randn(obs_sizes[i]) for i, agent in enumerate(self.agents)}

    def step(self, action_dict):
        # Perform a step of the environment for each agent
        observations = {}
        rewards = {}
        dones = {}
        infos = {}

        for agent, action in action_dict.items():
            obs_size = self.observation_space[agent].shape[0]
            observations[agent] = np.random.randn(obs_size)
            rewards[agent] = 0.0
            dones[agent] = False
            infos[agent] = {}

        dones["__all__"] = False

        return observations, rewards, dones, infos


class MyModelClass(FullyConnectedNetwork):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.fc1 = None
        self.fc2 = None
        self._setup_layers()

    def _setup_layers(self):
        self.fc1 = self._get_activation_fn()(self.fc1)
        self.fc2 = self._get_activation_fn()(self.fc2)


def train():
    ray.init()

    ModelCatalog.register_custom_model("my_model", MyModelClass)

    config = {
        "env": VariableSpaceEnv,
        "env_config": {
            "num_agents": 3,  # Specify the number of agents
            "episode_length": 100,  # Specify the episode length
        },
        "model": {
            "custom_model": "my_model",
        },
        "framework": "torch",
        "num_workers": 1,
        "train_batch_size": 1000,
        "sgd_minibatch_size": 100,
        "num_sgd_iter": 10,
        "lr": 0.001,
        "multiagent": {
            "policies": {
                agent: (None, Box(low=-1.0, high=1.0, shape=(10,)))  # Specify the action space for each agent
                for agent in ["agent_0", "agent_1", "agent_2"]  # Specify the names of the agents
            },
            "policy_mapping_fn": lambda agent_id: agent_id,  # Use the agent's name as its policy ID
            "policies_to_train": ["agent_0", "agent_1"],  # Only train the policy for agent 0
            "observation_fn": lambda obs: obs["obs"],  # Extract the observation for each agent
            "replay_mode": "independent",  # Enable parameter sharing by setting replay mode to independent
        },  # TODO: You must match the action space of each policy to that of each agent the environment
    }

    trainer = PPOTrainer(config=config)
    for i in range(10):
        result = trainer.train()
        print(result)

    ray.shutdown()


if __name__ == "__main__":
    env = VariableSpaceEnv({"num_agents": 3, "episode_length": 10})
    print(f"env.observation_space: {env.observation_space}")
    print(f"env.action_space: {env.action_space}")
    print(f"obs = {env.reset()}")
    print(f"env.observation_space: {env.observation_space}")
    print(f"env.action_space: {env.action_space}")

    train()

