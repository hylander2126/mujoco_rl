import numpy as np
import torch
from torch import nn

class Policy_Network(nn.Module):
    # The most basic network we can make is just a "Multi-Layer Perceptron" - often consists of 2 linear layers
    # that take the observation state as input and, in this case, outputs the mean (mu) and std (sigma) of the Gaussian policy.
    def __init__(self, input_dim, output_dim):
        super().__init__()

        hidden_space1 = 16 # 128 is common, let's keep it small for now
        hidden_space2 = 32 # Change if desired

        # For practice, let's make a simple feedforward network with two hidden layers and two outputs
        self.fc1 = nn.Linear(input_dim, hidden_space1) # First hidden layer
        self.fc2 = nn.Linear(hidden_space1, hidden_space2) # Second hidden layer
        self.policy_out_net = nn.Linear(hidden_space2, output_dim) # Output layer for mean (mu)

        # Activate with ReLu (or could use Tanh, sigmoid, etc.) - ReLu is efficient
        self.relu = nn.ReLU()

        self.eps = 1e-6 # Small rounding term
        
    def forward(self, obs):
        # Conditioned on observation, return the mean and std of the Gaussian policy
        # obs is observation from environment (a vector of state features)
        first_hidden = self.fc1(obs)
        first_active = self.relu(first_hidden) # Apply activation function
        second_hidden = self.fc2(first_active)
        second_active = self.relu(second_hidden) # Apply activation function
        action_out = self.policy_out_net(second_active) # Get mean (mu) from network

        return action_out # Don't constrain output here...


class REINFORCE():
    def __init__(self, obs_space_dims, act_space_dims, mu=-2.0, sigma=1.0):
        """ REINFORCE is the most basic policy gradient and deep RL algo. Uses MC sampling to estimate
        returns and update policy parameters. """
        self.mu             = mu
        self.sigma          = sigma # standard deviation for the Gaussian policy
        self.eps            = 1e-6 # Small term to prevent numerical instability in std

        ## Hyperparameters ##
        # Learning rate limits how huge the jumps between sequential steps are, to avoid overshooting.
        self.learning_rate  = 1e-3 # alpha or eta, depending on ref
        # Gamma limits rewards in long-horizon tasks, with ~1 being far-sighted and ~0 being near
        self.gamma          = 0.99 # Discount factor for future rewards. Close to 1 for now
        
        ## Variables to store episode data for policy update ##
        self.log_probs = [] # Store probability values of sampled actions
        self.rewards = [] # Store corresponding rewards
        
        # Input is 3D for pendulum, output is 2D (mean and std). 2*act_space because for each action dim, we need gaussian
        self.net = Policy_Network(input_dim=obs_space_dims, output_dim=2*act_space_dims)

        # This optimizer is just a nice wrapper for Gradient Descent. We initialize it to the network params
        # Basically, it will 'look at' the .grad attribute of net params, and use that to determine the 'update step'
        # for the network, changing weights and biases according to whatever loss function is defined.
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
    

    def sample_action(self, obs):
        """ Sample action from policy. This is a normal distribution with parameters mu, sigma. x is action."""
        # print("FAILED JUST AFTER THIS")

        ## Feed observation to network ## 
        mu_sigma_out = self.net(torch.tensor(obs)) # Make sure obs is a tensor for network

        mu = mu_sigma_out[0] # First output is mean (mu)
        sigma = torch.exp(torch.clamp(mu_sigma_out[1], -20, 2)) + self.eps # Second output is std (sigma), but can't have negative sigma...
        
        # Create Gaussian dist. with mean and std from net. This dist. is used to create a stochastic (probabilistic) 'map' from which
        # we will sample from in the very next line
        distrib = torch.distributions.Normal(loc=mu, scale=sigma) # Just a normal dist, syntax'ed for torch        

        # Yup, here we choose action from distribution. (but keep as torch tensor for backprop later)
        action = distrib.sample()

        # We have to 'clip' to action limits [-2, 2], but then we have to 'correct' log prob to account for this new distribution
        squashed = torch.tanh(action) # Squash to [-1, 1]
        action = 2*squashed # Scale to [-2, 2]
        
        ## Log Probability of action under that policy ##
        # This log prob is used to update policy. Acts like a 'weight' on how much to update policy, based on how good return was.
        # "How likely was this action under the policy output from the net?" If unlikely, but good reward, update policy 'harder' w bigger step.
        log_prob = distrib.log_prob(action) - torch.log(1 - squashed.pow(2) + 1e-6)
        self.log_probs.append(log_prob)

        return action
    
    def update_policy(self):
        """ Update policy parameters (theta) which is just mu and std in this case. Use gradient ascent.
            This is the core of REINFORCE, other RL algos use a different update rule, but same concpt. 
        """

        # First calculate discounted returns (G_t) from cumulative rewards by looping through recorded rewards backwards.
        # Discounted rewards conceptually is that future rewards are worth less than immediate rewards, so *gamma to 'discount' those future rewards.
        rewards_ordered = []
        reward_cumulative_running = 0
        
        # self.rewards.reverse() # backwards pass thru rewards to calculate discounted return at each t
        # bkwd_r = self.rewards
        for r in self.rewards[::-1]:
            reward_cumulative_running = r + self.gamma * reward_cumulative_running
            rewards_ordered.insert(0, reward_cumulative_running) # Insert at front to maintain correct order (or flip afterwards!)

        # Calculate policy LOSS (for NN) - REINFORCE loss rule is to negate log prob, weighted by disc. return for each t step
        L = 0
        logprob_and_returns = zip(self.log_probs, rewards_ordered) # Pair log probabilities with their corresponding returns
        for lp, g in logprob_and_returns:
            L -= lp * g # negative because gradient ascent, to maximize returns

        ## Backprop time! The optimizer is just Gradient Descent, but a nicer 'torch' version ##
        # Clear old gradients (if any) from previous update
        self.optimizer.zero_grad()
        # Compute the gradient dL/dtheta of Loss wrt network parameters theta
        L.backward() # This stores inside the parameters .grad attribute. the optimizer 
        self.optimizer.step() # Recall optimizer looks @ net params, so this updates weights and biases!

        # Empty the episode-centric variables to prepare for next episode
        self.log_probs = []
        self.rewards = []