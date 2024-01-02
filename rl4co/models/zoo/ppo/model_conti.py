from rl4co.envs import RL4COEnvBase
from rl4co.models.rl import PPOContinuous
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.models.zoo.ppo.policy_conti import PPOContiPolicy


class PPOContinuousModel(PPOContinuous):
    """PPO Model based on Proximal Policy Optimization (PPO).

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        critic: Critic to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        critic_kwargs: Keyword arguments for critic
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        opponent: object or None,
        policy: PPOContiPolicy = None,
        critic: CriticNetwork = None,
        policy_kwargs: dict = {},
        critic_kwargs: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = PPOContiPolicy(env.name, **policy_kwargs)

        if critic is None:
            critic = CriticNetwork(env.name, **critic_kwargs)

        super().__init__(env, opponent, policy, critic, **kwargs)
