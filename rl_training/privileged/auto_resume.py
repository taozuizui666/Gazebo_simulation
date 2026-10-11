"""Select a complete saved model from the most recent multi-map run."""

from dataclasses import asdict
import json
import re

from stable_baselines3.common.save_util import load_from_pkl
from stable_baselines3.common.vec_env import VecNormalize

from rl_training.common.model_io import load_model


def normalization_path(model):
    if model.name == 'best_model.zip':
        return model.parent / 'best_vecnormalize.pkl'
    match = re.fullmatch(r'rl_policy_(\d+)_steps.zip', model.name)
    if match:
        return model.parent / f'rl_policy_vecnormalize_{match[1]}_steps.pkl'
    return model.parent / 'vecnormalize.pkl'


def check_checkpoint(path):
    """Deserialize both files before selecting them; existence alone is insufficient."""
    model = load_model(path, device='cpu')
    normalizer = load_from_pkl(normalization_path(path))
    if not isinstance(normalizer, VecNormalize):
        raise ValueError('归一化文件不是 VecNormalize。')
    if (normalizer.observation_space != model.observation_space
            or normalizer.action_space != model.action_space):
        raise ValueError('模型与归一化文件的输入/动作空间不匹配。')
    match = re.fullmatch(r'rl_policy_(\d+)_steps.zip', path.name)
    if match and int(match[1]) != model.num_timesteps:
        raise ValueError('检查点文件名与模型步数不匹配。')
    return model.num_timesteps


def resolve_resume(args, enabled):
    if not enabled:
        if args.resume is not None:
            check_checkpoint(args.resume)
        return args.resume
    # Run names start with local YYYYMMDD_HHMMSS, so sort by run creation time.
    runs = sorted(args.output_dir.glob('PPO_privileged_*'), reverse=True)
    for directory in runs:
        if not directory.is_dir() or directory.is_symlink():
            continue
        try:
            metadata = json.loads((directory / 'run_config.json').read_text())
        except (OSError, ValueError):
            continue
        if metadata.get('observation_mode') != 'privileged' or not metadata.get('multi_map'):
            continue
        # Final/interrupted saves represent the end of this run. Otherwise use
        # periodic checkpoints by step, never a best-reward model from an earlier step.
        candidates = [directory / name for name in ('rl_policy_final.zip', 'rl_policy_interrupted.zip')]
        checkpoints = [p for p in directory.glob('rl_policy_*_steps.zip')
                       if re.fullmatch(r'rl_policy_\d+_steps.zip', p.name)]
        candidates += sorted(checkpoints, key=lambda p: int(p.stem.split('_')[-2]), reverse=True)
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                steps = check_checkpoint(candidate)
            except Exception as error:
                print(f'跳过不完整或无法读取的续训文件 {candidate}：{type(error).__name__}: {error}', flush=True)
                continue
            print(f'自动续训：{candidate}（累计 {steps} 步）', flush=True)
            return candidate
    print(f'没有可用的多地图检查点，使用 RESUME_MODEL：{args.resume or "None（从零训练）"}', flush=True)
    if args.resume is not None:
        check_checkpoint(args.resume)
    return args.resume


def validate_resume(args):
    if args.resume is None:
        return
    from rl_training.common import config
    from rl_training.common.observations import observation_specification
    saved = json.loads((args.resume.parent / 'run_config.json').read_text())
    vehicle = config.vehicle_config(args.config_header)
    if saved.get('schema_version') not in (2, 3) or saved.get('observation_mode') != 'privileged':
        raise ValueError('续训需要本版特权模型。')
    if saved.get('observation_spec') != observation_specification('privileged', vehicle):
        raise ValueError('续训模型观测规格与当前配置不同；请检查配置或关闭 AUTO_RESUME 指定模型。')
    if saved.get('vehicle') != asdict(vehicle) and not args.resume_allow_vehicle_change:
        raise ValueError('续训模型车辆参数与当前配置不同。')
    if saved.get('environment', {}).get('step_sim_sec') != args.environment['step_sim_sec']:
        raise ValueError('续训不能改变 STEP_SIM_TIME_SEC。')
    if saved.get('ppo', {}).get('POLICY_NET_ARCH') != args.ppo['POLICY_NET_ARCH']:
        raise ValueError('续训不能改变 POLICY_NET_ARCH。')
