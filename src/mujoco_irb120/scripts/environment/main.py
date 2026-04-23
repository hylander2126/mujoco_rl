#!/usr/bin/env python3
"""Test script to validate env.py wrapper functionality."""

import os
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "src"))

print(os.getcwd())

from mujoco_irb120.scripts.environment.env import IRB120Env


def test_env_wrapper():
    """Run env.py through all major features."""
    print("Initializing RobotEnv...")
    env = IRB120Env(object_id=0, render_mode=None)
    
    print("Testing reset()...")
    obs, info = env.reset()
    print(f"  ✓ Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    
    print("Testing step() with random action...")
    action = env.action_space.sample() if hasattr(env, 'action_space') else [0.0] * 6
    obs, done, info = env.step(action)
    print(f"  ✓ Step executed. Done: {done}")
    
    print("Testing state access...")
    print(f"  ✓ Info: {info}")
    
    print("Testing multiple steps...")
    for i in range(10):
        obs, done, info = env.step(action)
    print(f"  ✓ 10 steps completed")
    
    print("Testing close()...")
    env.close()
    print("  ✓ Environment closed cleanly")
    
    print("\n✅ All env.py features validated!")

if __name__ == "__main__":
    test_env_wrapper()