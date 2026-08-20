"""PPO runner configuration for the squat-before-box task.

The observation vector is much larger than in the plain velocity task because
of the flattened depth image. We keep the MLP but widen the first layer.
For serious camera-based training, swap the actor/critic for a CNN backbone
(see README).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class SquatBeforeBoxPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 200
    experiment_name = "g1_squat_before_box"
    empirical_normalization = True  # helps a lot with the wide-range depth obs

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # First layer wide enough to absorb the flattened 48x48 depth (=2304)
        # plus proprio history.
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
