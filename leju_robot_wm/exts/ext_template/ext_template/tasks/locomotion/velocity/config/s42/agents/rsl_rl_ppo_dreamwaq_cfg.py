from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from dataclasses import MISSING


@configclass
class CENetCfg:
    class_name: str = "CENet"
    latent_dim: int = MISSING
    encoder_hidden_dims: list[int] = MISSING
    decoder_hidden_dims: list[int] = MISSING
    activation: str = MISSING
    learning_rate: float = MISSING
    max_grad_norm: float = MISSING
    beta: float = MISSING


@configclass
class RslRlPpoAlgorithmCENetCfg(RslRlPpoAlgorithmCfg):
    latent_to_critic: bool = MISSING
    obs_history_len: int = MISSING


@configclass
class KuavoS42RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 50
    experiment_name = "s42/s42_rough_dreamwaq_v0"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCENetCfg(
        class_name="PPO_DreamWaq",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        latent_to_critic=False,
        obs_history_len=5,
    )

    cenet = CENetCfg(
        latent_dim=35,
        encoder_hidden_dims=[512, 256],
        decoder_hidden_dims=[256, 512],
        activation="elu",
        learning_rate=1e-3,
        max_grad_norm=1.0,
        beta=1e-5,
    )


@configclass
class KuavoS42FlatPPORunnerCfg(KuavoS42RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 15000
        self.experiment_name = "s42/s42_flat_dreamwaq_v0"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
