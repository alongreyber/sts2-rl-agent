"""Explicit schema lookup; never reinterpret checkpoint dimensions silently."""
import importlib


def schema(version):
    if version not in (2, 3):
        raise ValueError('Expected expanded observation version 2 or 3')
    return importlib.import_module(f'sts2_env.gym_env.combat_observation_v{version}')


def version_for_shape(shape):
    if shape == (217,): return 1
    for version in (2, 3):
        if shape == (schema(version).OBS_SIZE,): return version
    raise ValueError(f'Unknown combat checkpoint observation shape: {shape}')
