"""Official RSL-RL PPO configuration with an asymmetric actor-critic."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BPXFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 2026
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "bpx_flat"
    clip_actions = 100.0

    # Actor: policy group only = 45-D. Critic: policy + privileged = 48-D.
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "privileged"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BPXRoughPPORunnerCfg(BPXFlatPPORunnerCfg):
    """Keep checkpoint-compatible networks and log mixed-terrain runs separately."""

    experiment_name = "bpx_rough"


@configclass
class BPXStairsPPORunnerCfg(BPXRoughPPORunnerCfg):
    """Use a dedicated log directory for stair-focused training."""

    experiment_name = "bpx_stairs"


@configclass
class BPXRoughTeacherPPORunnerCfg(BPXFlatPPORunnerCfg):
    """PPO teacher: both actor and critic see simulated terrain and velocity."""

    experiment_name = "bpx_rough_teacher"
    obs_groups = {
        "policy": ["policy", "privileged", "terrain"],
        "critic": ["policy", "privileged", "terrain"],
    }


@configclass
class BPXStairsTeacherPPORunnerCfg(BPXRoughTeacherPPORunnerCfg):
    experiment_name = "bpx_stairs_teacher"


@configclass
class BPXRoughBlindDistillationRunnerCfg(BPXFlatPPORunnerCfg):
    """Official RSL-RL 3.0.1 DAgger-style distillation with a history MLP.

    Isaac Lab 2.3.0 has no distillation config dataclasses, so pass the
    native RSL-RL policy/algorithm dictionaries through the existing runner.
    This is an action-imitation baseline, not the paper's TCN/latent loss.
    """

    class_name = "DistillationRunner"
    experiment_name = "bpx_rough_blind"
    obs_groups = {
        "policy": ["proprio_history"],
        "teacher": ["policy", "privileged", "terrain"],
    }
    policy: dict = {
        "class_name": "StudentTeacher",
        "init_noise_std": 0.1,
        "noise_std_type": "scalar",
        "student_obs_normalization": False,
        "teacher_obs_normalization": False,
        "student_hidden_dims": [512, 256, 128],
        "teacher_hidden_dims": [256, 256, 128],
        "activation": "elu",
    }
    algorithm: dict = {
        "class_name": "Distillation",
        "num_learning_epochs": 5,
        # Feed-forward history MLP: update every stored step, no RNN BPTT.
        "gradient_length": 1,
        "learning_rate": 1.0e-4,
        "max_grad_norm": 1.0,
        "loss_type": "mse",
        "optimizer": "adam",
    }


@configclass
class BPXStairsBlindDistillationRunnerCfg(BPXRoughBlindDistillationRunnerCfg):
    experiment_name = "bpx_stairs_blind"
