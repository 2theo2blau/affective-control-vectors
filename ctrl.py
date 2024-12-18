import torch
import torch.nn as nn
import torch.nn.Functional as F

class AffectiveControlVectors(nn.Module):
    def __init__(self, embedding_dim, num_affective_states):
        super(AffectiveControlVectors, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_affective_states = num_affective_states

        # Initialize control vectors for each affective state
        self.control_vectors = nn.Parameter(torch.randn(num_affective_states, embedding_dim))

        # Reinforcement learning parameters
        self.affective_state_rewards = torch.zeros(num_affective_states)
        self.affective_state_usage_count = torch.zeros(num_affective_states)

        # Exploration parameters
        self.exploration_rate = 0.1 # Probability of selecting a random affective state

    def forward(self, hidden_states, affective_state_index, alpha=1.0):
        """
        Adjusts the hidden states based on the selected affective state vector.

        Parameters:
        - hidden_states (torch.Tensor): the input hidden states from the language model
        - affective_state_index (int, optional): Index of the desired affective stte. If None, selects based on exploration rate.
        - alpha (float): The scaling factor of the control vector

        Returns:
        - adjusted_hidden_states (torch.Tensor): The modified hidden state
        """

        # Determine which affective state to use
        if affective_state_index is None:
            # Select affective state based on exploration vs exploitation
            if torch.rand(1).item() < self.exploration_rate:
                # Exploration: Randomly select an affective state
                affective_state_index = torch.randint(0, self.num_affective_states, (1,)).item()
            else:
                # Exploitation: Select the affective state with the highest average reward
                average_rewards = self.affective_state_rewards / (self.affective_state_usage_count + 1e-5)
                affective_state_index = torch.argmax(average_rewards).item()

        # Update usage count for the selected affective state
        self.affective_state_usage_count[affective_state_index] += 1

        # Retrieve the control vector for the selected affective state
        control_vector = self.control_vectors[affective_state_index]

        # Expand control vector to match hidden states shape
        control_vector = control_vector.unsqueeze(0).expand_as(hidden_states)

        # Adjust the hidden states using the control vector
        adjusted_hidden_states = hidden_states + alpha * control_vector

        return adjusted_hidden_states
    
    def update_rewards(self, affective_state_index, reward):
        """
        Updates the reward value for a specific affective state based on feedback.

        Parameters:
        - affective_state_index (int): The index of the affective state to update.
        - reward (float): The reward value to add for the affective state.
        """
        self.affective_state_rewards[affective_state_index] += reward

    def normalize_control_vectors(self):
        """
        Normalizes the control vectors to ensure stability in their magnitude.
        """
        with torch.no_grad():
            self.control_vectors = nn.Parameter(F.normalize(self.control_vectors, p=2, dim=1))

    def anneal_exploration(self, decay_factor=0.99):
        """
        Gradually reduces the exploration rate over time.

        Parameters:
        - decay_factor (float): The rate at which to decay the exploration parameter.
        """
        self.exploration_rate *= decay_factor
        self.exploration_rate = max(self.exploration_rate, 0.01) # Maintain a minimum exploration rate for diversity

# Example usage
if __name__ == "__main__":
    embedding_dim = 768
    num_affective_states = 5

    control_vectors_module = AffectiveControlVectors(embedding_dim, num_affective_states)
    hidden_states = torch.randn(1, embedding_dim)

    # Adjust hidden states with a selected affective state
    adjusted_hidden_states = control_vectors_module(hidden_states, affective_state_index=2, alpha=0.5)

    # Simulate a reward update
    control_vectors_module.update_rewards(affective_state_index=2, reward=1.0)

    # Normalize control vectors periodically
    control_vectors_module.normalize_control_vectors()

    # Anneal exploration rate over time
    control_vectors_module.anneal_exploration()

    