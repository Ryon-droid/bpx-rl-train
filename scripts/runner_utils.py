"""Select native RSL-RL runners and export their inference-only networks."""

from policy_contract import actor_input_dim, export_basename, write_export_contract


def make_runner(env, agent_cfg, log_dir=None):
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    runners = {"OnPolicyRunner": OnPolicyRunner, "DistillationRunner": DistillationRunner}
    config = agent_cfg.to_dict()
    name = config.get("class_name", "OnPolicyRunner")
    if name not in runners:
        raise ValueError(f"Unsupported runner: {name}")
    return runners[name](env, config, log_dir=log_dir, device=agent_cfg.device)


def export_runner_policy(runner, task, output_dir):
    from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

    policy = runner.alg.policy
    if hasattr(policy, "student"):
        normalizer = policy.student_obs_normalizer
        network = policy.student
    else:
        normalizer = policy.actor_obs_normalizer
        network = policy.actor
    if network[0].in_features != actor_input_dim(task):
        raise ValueError("Actor input dimension does not match the selected task's export contract")
    basename = export_basename(task)
    export_policy_as_jit(policy, normalizer=normalizer, path=output_dir, filename=f"{basename}.pt")
    export_policy_as_onnx(policy, normalizer=normalizer, path=output_dir, filename=f"{basename}.onnx")
    write_export_contract(task, output_dir)
    return basename


def load_runner_checkpoint(runner, checkpoint, *, teacher_initialization=False, load_optimizer=True):
    """Do not accidentally evaluate a randomly initialized blind student."""
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved["model_state_dict"]
    has_student = any(key.startswith("student.") for key in state)
    has_actor = any(key.startswith("actor.") for key in state)
    is_student_runner = hasattr(runner.alg.policy, "student")
    if teacher_initialization:
        if not is_student_runner or not has_actor or has_student:
            raise ValueError("--teacher_checkpoint must be a PPO *_teacher checkpoint")
    elif is_student_runner and not has_student:
        raise ValueError("A *_blind task needs a trained student checkpoint; initialize distillation using train --teacher_checkpoint")
    elif not is_student_runner and not has_actor:
        raise ValueError("A PPO task needs an actor-critic checkpoint")
    del saved, state
    return runner.load(checkpoint, load_optimizer=load_optimizer)
