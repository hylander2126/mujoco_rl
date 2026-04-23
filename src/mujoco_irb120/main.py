#!/usr/bin/env python3
"""Test script to validate env.py wrapper functionality."""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from mujoco_irb120.scripts.environment.env import IRB120Env


def test_env_wrapper():
    """Run env.py through all major features."""
    print("Initializing RobotEnv...")
    env = IRB120Env(object_id=0, use_viewer=False, record_video=False)
    
    print("Testing reset()...")
    obs = env.reset()
    print(f"  ✓ Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    
    print("Testing step() with random action...")
    action = env.action_space.sample() if hasattr(env, 'action_space') else [0.0] * 6
    obs, reward, done, info = env.step(action)
    print(f"  ✓ Step executed. Reward: {reward}, Done: {done}")
    
    print("Testing state access...")
    print(f"  ✓ Robot EE pose: {env.get_ee_pose()}")
    print(f"  ✓ F/T sensor: {env.get_wrench()}")
    
    print("Testing multiple steps...")
    for i in range(10):
        obs, reward, done, info = env.step(action)
    print(f"  ✓ 10 steps completed")
    
    print("Testing close()...")
    env.close()
    print("  ✓ Environment closed cleanly")
    
    print("\n✅ All env.py features validated!")

if __name__ == "__main__":
    test_env_wrapper()