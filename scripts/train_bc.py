from __future__ import annotations

from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# from models.policy import TinyVLAPolicy

"""
Behavior Cloning (BC) for imitation learning as supervised learning problem.
Given expert trajectories (observation-action pairs), policy network is trained to IMITATE expert.

Expert data is .npz archive with keys: 
# images, (N, 128, 128, 3) RGB
# states, (N, S=24)
# actions, (N, 6 joints)
# instructions, (N, ) strings
# cube_color, (N, ) strings
# episode_idx, (N, ) int
# step_idx, (N, ) int
# success, (N, ) bool
# record_stride, (1, ) = 1
# sim_timestep, (1, ) = 0.001 seconds
# max_sim_time (1, ) = 5.0 seconds

obs, actions: shape (N * L,) + S 
# where N = # episodes, L = episode length, and S is envrionment obs/action space (1, for discrete space)
"""

def train_bc(
    dataset_path,
    checkpoint_dir,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    train_split,
    seed,
):
    dataset_path = Path(dataset_path)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    data = np.load(dataset_path, allow_pickle=True) # Allows loading string arrays

    # Most important arrays are: observations (images and states) and target (action)
    # We actually don't care about success, because the expert theoretically doesn't make them, or prunes them out...

    # ==== 1. First try with just states (state_t -> delta_action_t)
    states = data["states"] # shape (N, S)
    expert_joint_targets = data["actions"] # shape (N, A), absolute joint targets from the expert
    current_joint_positions = states[:, : expert_joint_targets.shape[1]]
    actions = expert_joint_targets - current_joint_positions # Train the policy to predict joint deltas
    print(f"States shape: {states.shape}, Joint-delta actions shape: {actions.shape}")

    # ==== 2. Split train/test sets. Want to generalize to new episodes, not new timesteps within the same episode. SO:
    # Do NOT randomly split individual timesteps, consecutive times are correlated. Instead, split by episode.
    episode_idx = data["episode_idx"] # shape (N, )
    # The following gets distinct episode IDs from data. ([0, 0, 0, 1, 1, 2, 2, 2, ...] -> [0, 1, 2, ...])
    # Normally, episodes might have different lengths, so this simply gets the unique IDs no repeats.
    unique_episodes = np.unique(episode_idx) 
    np.random.shuffle(unique_episodes) # Shuffle episode indices for random train/test split
    
    # Split shuffled episodes into train and validation sets
    validation_frac = 1.0 - train_split
    n_validation = int(len(unique_episodes) * validation_frac)
    if len(unique_episodes) > 1 and validation_frac > 0.0:
        n_validation = max(1, n_validation)
    validation_episodes = unique_episodes[:n_validation] # Index of episodes for validation
    train_episodes = unique_episodes[n_validation:] # Index of episodes for training

    # Need the indices representing each split. NOT Boolean masks.
    train_indices = np.where(np.isin(episode_idx, train_episodes))[0] # Indices for training samples
    validation_indices = np.where(np.isin(episode_idx, validation_episodes))[0] # Indices for validation samples

    print(f"Training indices: {len(train_indices)}, Validation indices: {len(validation_indices)}")

    # ==== 3. Normalize states and actions. Important for training stability across any Learning architecture.
    state_mean = states[train_indices].mean(axis=0)
    state_std = states[train_indices].std(axis=0) + 1e-6 # for no divide by zero

    action_mean = actions[train_indices].mean(axis=0)
    action_std = actions[train_indices].std(axis=0) + 1e-6

    def normalize(x, mean, std):
        """Z-score normalization to scale data for mean=0 and std=1."""
        return (x - mean) / std
    def unnormalize(z, mean, std):
        return (z * std) + mean
    
    # We do this so that: during TRAINING, we compare NORMALIZED predicted output to NORMALIZED expert action (target)
    # And during DEPLOYMENT, we can un-normalize the network output.

    # ==== 4. Make PyTorch Dataset for nice and easy training.
    # Best done by creating a custom dataset class which inherits from torch.utils.data.Dataset. This class will
    # load the data, apply normalization, and return (state, action) pairs for training.
    class StateBCDataset(Dataset):
        def __init__(self, states, actions, indices):
            self.states = states
            self.actions = actions
            self.indices = indices

        def __len__(self): # torch uses this to know how many samples are in the dataset
            return len(self.indices)
        
        def __getitem__(self, idx): # torch uses this to get a single sample (state, action) pair
            i = self.indices[idx] # NOTE: what does this do?
            s = normalize(self.states[i], state_mean, state_std) # Normalize state
            a = normalize(self.actions[i], action_mean, action_std) # Normalize action
            return {
                "state": torch.tensor(s, dtype=torch.float32), # Convert to PyTorch tensor
                "action": torch.tensor(a, dtype=torch.float32) # Convert to PyTorch tensor
            }
    
    # Create train and validation datasets
    train_dataset = StateBCDataset(states, actions, train_indices)
    validation_dataset = StateBCDataset(states, actions, validation_indices)

    # Then dataloader for batching and shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, # Hyperparameter: number of samples per batch
        shuffle=True # Shuffle training data for better generalization
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size, # Hyperparameter: number of samples per batch
        shuffle=False # No need to shuffle validation data
    )

    # ==== 5. Write the simplest policy network
    state_dim = states.shape[1] # State dimension
    action_dim = actions.shape[1] # Action dimension

    # Classic feedforward MLP with 2 hidden layers, ReLU, output with no activation
    class StatePolicy(nn.Module):
        def __init__(self, state_dim, action_dim): 
            super().__init__() # Call parent constructor
            self.net = nn.Sequential(
                nn.Linear(state_dim, 256), # Input layer to first hidden layer. 256 is common layer size
                nn.ReLU(), # Activation function for non-linearity

                nn.Linear(256, 256), # First hidden layer to second hidden layer
                nn.ReLU(), # Activation function for non-linearity

                nn.Linear(256, action_dim) # Second hidden layer to output layer. Output is action_dim (6 for 6 joints)
            )
        def forward(self, state): # single forward pass through the network
            return self.net(state)
    
    # And instantiate the network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available for faster training
    model = StatePolicy(state_dim, action_dim).to(device) # Move model to device (GPU or CPU)

    # ==== 6. One training step
    # One batch contains states and expert actions
    # The following creates an iterator over the DataLoader, grab the first batch. Acts as smoke test for one batch 
    # of training data to check shapes, and run one forward pass.
    batch = next(iter(train_loader))
    state = batch["state"].to(device) # Move batch to device
    action = batch["action"].to(device) # Move batch to device

    pred_action = model(state) # Forward pass: predict action from state

    print(f"These should match and be (batch_size, action_dim): {pred_action.shape}, {action.shape}") # Should both be (batch_size, action_dim)

    # And write the loss. Here, let's use MSE loss, common for continuous action spaces. Compare pred action
    # to expert action (target).
    loss_fn = nn.MSELoss() # Mean Squared Error loss for regression
    loss = loss_fn(pred_action, action) # Compute loss between predicted and expert actions
    print(f"Initial loss (should be > 0 and probably large): {loss.item()}")


    # ==== 7. Training loop. Now we need an optimizer to update network weights based on loss.
    optimizer = torch.optim.AdamW(
        model.parameters(), # Tell optimizer which parameters to update
        lr=learning_rate, # Learning rate hyperparameter
        weight_decay=weight_decay # L2 regularization to prevent overfitting. Unique to AdamW (decoupled weight decay) optimizer.
    )

    def run_epoch(model, loader, optimizer=None): # For one epoch (batch)
        is_training = optimizer is not None # If optimizer is provided, we are in training mode

        if is_training:
            model.train() # Set model to training mode (enables dropout, batchnorm, etc.)
        else:
            model.eval() # Set model to evaluation mode (disables dropout, batchnorm, etc.)
        
        total_loss = 0.0 # Accumulate loss over batches, initialize to zero
        total_count = 0  # number of samples (sample is one state-action pair) for averaging loss since batch sizes may vary

        for batch in loader: # recall loader is just a data structure that gives batches of data from dataset (here, the training set or validation set, depending on train/eval mode)
            state = batch["state"].to(device) # Move batch to device
            action = batch["action"].to(device) # Move batch to device

            with torch.set_grad_enabled(is_training): # Only compute gradients if training (so we can also use func for evaluation)
                pred_action = model(state) # Forward pass: predict action from state
                loss = loss_fn(pred_action, action) # Compute loss between predicted and expert actions

                # And if training, backpropagate and update model parameters
                if is_training:
                    optimizer.zero_grad() # Clear previous gradients
                    loss.backward() # Backpropagate to compute gradients
                    optimizer.step() # Update model parameters based on gradients
            
            batch_size = state.shape[0]             # Number of samples in this batch (first dimension of state tensor)
            total_loss += loss.item() * batch_size  # Add total loss. Multiply current batch loss by size of batch for total loss for ALL samples.
            total_count += batch_size               # Accumulate total sample count
            
        if total_count == 0:
            return None
        return total_loss / total_count # Return average loss per sample for this epoch

    # Then train
    for epoch in range(epochs): # Hyperparameter: number of epochs to train for
        train_loss = run_epoch(model, train_loader, optimizer) # Run training epoch
        # Now evaluate on validation set without updating weights
        validation_loss = run_epoch(model, validation_loader, optimizer=None) # Run validation epoch (no optimizer, so no weight updates)
        if validation_loss is None:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Validation Loss = skipped (no validation samples)")
        else:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Validation Loss = {validation_loss:.4f}")

    # And now we have behavior cloning! Let's save the model and normalization parameters for deployment.
    ckpt = {
        "model": model.state_dict(), # Save model weights
        "model_state_dict": model.state_dict(), # Same naming convention as the other project scripts
        "policy_type": "state_only",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": 256,
        "state_mean": torch.tensor(state_mean, dtype=torch.float32), # Save state normalization mean
        "state_std": torch.tensor(state_std, dtype=torch.float32), # Save state normalization std
        "action_mean": torch.tensor(action_mean, dtype=torch.float32), # Save action normalization mean
        "action_std": torch.tensor(action_std, dtype=torch.float32), # Save action normalization std
        "action_mode": "joint_delta",
        "record_stride": int(data["record_stride"]) if "record_stride" in data.files else None,
        "ft_bias_enabled": bool(data["ft_bias_enabled"]) if "ft_bias_enabled" in data.files else None,
        "ft_bias_samples": int(data["ft_bias_samples"]) if "ft_bias_samples" in data.files else None,
        "dataset_path": str(dataset_path),
    }
    checkpoint_path = checkpoint_dir / "bc_only_states.pt"
    torch.save(ckpt, checkpoint_path) # Save checkpoint to file
    print(f"Saved state-only BC checkpoint to {checkpoint_path}")
    return checkpoint_path

def main():
    # For now, just test loading the dataset and print shapes
    dataset_path = Path("outputs/rollouts/sim_vla_rollouts.npz")
    train_bc(
        dataset_path=dataset_path,
        checkpoint_dir=Path("outputs/checkpoints"),
        epochs=10,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-5,
        train_split=0.9,
        seed=7,
    )

if __name__ == "__main__":
    main()
