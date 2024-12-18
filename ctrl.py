import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

class NeuralAbstractionLayer(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(NeuralAbstractionLayer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model.transformer(**inputs)
        hidden_states = outputs.last_hidden_state
        return hidden_states, inputs['input_ids']
    
    def decode(self, hidden_states):
        logits = self.model.lm_head(hidden_states)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_ids = torch.argmax(probabilities, dim=-1)
        text = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        return text
    
# Representation Reading Module
class RepReader(nn.Module):
    def __init__(self, embedding_dim, num_affective_states):
        super(RepReader, self).__init__()
        self.affective_classifier = nn.Linear(embedding_dim, num_affective_states)

    def forward(self, hidden_states):
        pooled_rep = hidden_states.mean(dim=1)
        affective_logits = self.affective_classifier(pooled_rep)
        return affective_logits
    
    def adjust_representation(self, hidden_states, affective_state_indices, alpha=1.0):
        adjustment = self.control_vectors[affective_state_indices].unsqueeze(1)
        adjusted_rep = hidden_states + alpha * adjustment
        return adjusted_rep

# Representation Controller Module
class RepController(nn.Module):
    def __init__(self, embedding_dim, num_affective_states):
        super(RepController, self).__init__()
        self.control_vectors = nn.Parameter(torch.randn(num_affective_states, embedding_dim))

    def adjust_representation(self, hidden_states, affective_state_indices, alpha=1.0):
        adjustment = self.control_vectors[affective_state_indices].unsqueeze(1)
        adjusted_rep = hidden_states + alpha * adjustment
        return adjusted_rep
    
# Training Module
class AffectiveRLTrainer:
    def __init__(self, model_name, embedding_dim, num_affective_states, learning_rate=1e-4, gamma=0.99):
        self.neural_layer = NeuralAbstractionLayer(model_name)
        self.rep_reader = RepReader(embedding_dim, num_affective_states)
        self.rep_controller = RepController(embedding_dim, num_affective_states)

        self.optimizer = optim.Adam(list(self.rep_reader.parameters()) + list(self.rep_controller.parameters()), lr=learning_rate)
        self.gamma = gamma

    def train_step(self, text, target_affective_state, reward_function):
        hidden_states, input_ids = self.neural_layer.encode(text)

        # use representation reading to estimate current affective state
        current_affective_logits = self.rep_reader(hidden_states)
        current_affective_state = torch.argmax(current_affective_logits, dim=-1)

        # use representation control to adjust representation towards target state
        adjusted_hidden_states = self.rep_controller.adjust_representation(hidden_states, target_affective_state)

        # decode adjusted representation
        adjusted_text = self.neural_layer.decode(adjusted_hidden_states)

        # calculate reward based on adjusted text
        reward = reward_function(adjusted_text, target_affective_state)

        # compute loss and perform backpropagation
        loss = -reward # Policy gradient: maximize reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, dataset, reward_function, num_epochs=10):
        for epoch in range(num_epochs):
            for text, target_affective_state in dataset:
                self.train_step(text, target_affective_state, reward_function)

# Reward Function
class RewardFunction:
    def __init__(self, target_affective_states):
        self.target_affective_states = target_affective_states

    def __call__(self, generated_text, target_affective_state):
        # Calculate reward based on how closely the generated text matches the desired affective state
        reward = self.calculate_reward(generated_text, target_affective_state)
        return reward
    
    def calculate_reward(self, generated_text, target_affective_state):
        # Specific reward logic here based on sentiment analysis, coherence, etc.
        return torch.tensor(1.0) # to implement later
    

# Training loop
if __name__ == "__main__":
    embedding_dim = 768
    num_affective_states = 8
    learning_rate = 1e-4
    gamma = 0.99
    model_name = 'gpt2' # replace with mistral model when code is updated to support it
    reward_function = RewardFunction(target_affective_states=num_affective_states)

    trainer = AffectiveRLTrainer(model_name, embedding_dim, num_affective_states, learning_rate, gamma)

    dataset = [
        ("this is a sad story.", 2)
        ("I feel so happy today", 5)
        # placeholders -- more samples to come
    ]

    trainer.train(dataset, reward_function, num_epochs=10)