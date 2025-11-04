import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import faiss

class PatternNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super(PatternNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, input_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.relu(self.fc1(lstm_out[:, -1, :]))
        return self.fc2(out)

class VectorMemoryManager:
    def __init__(self, embedding_dim: int = 384):  # SentenceTransformer default dim
        self.dimension = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.memory_texts = []
        
    def add_memory(self, text: str, embedding: np.ndarray):
        """Add a new memory with its embedding"""
        self.index.add(embedding.reshape(1, -1))
        self.memory_texts.append(text)
        
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar memories"""
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.memory_texts):  # Valid index check
                results.append((self.memory_texts[idx], float(dist)))
        return results

    def get_size(self) -> int:
        """Get number of stored memories"""
        return len(self.memory_texts)

class DecisionNetwork(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(DecisionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.softmax(self.fc3(x), dim=1)

class SelfLearningModule:
    def __init__(self, embedding_dim: int = 384):
        self.pattern_net = PatternNetwork(embedding_dim)
        self.decision_net = DecisionNetwork(embedding_dim, num_classes=5)  # 5 decision types
        self.memory_buffer = deque(maxlen=1000)  # Store recent experiences
        self.optimizer_pattern = torch.optim.Adam(self.pattern_net.parameters())
        self.optimizer_decision = torch.optim.Adam(self.decision_net.parameters())
        
    def train_on_batch(self, embeddings: torch.Tensor, labels: torch.Tensor = None):
        """Train networks on a batch of data"""
        # Pattern recognition training
        self.pattern_net.train()
        self.optimizer_pattern.zero_grad()
        pred_patterns = self.pattern_net(embeddings)
        pattern_loss = nn.MSELoss()(pred_patterns, embeddings)
        pattern_loss.backward()
        self.optimizer_pattern.step()
        
        # Decision network training (if labels provided)
        if labels is not None:
            self.decision_net.train()
            self.optimizer_decision.zero_grad()
            decisions = self.decision_net(embeddings.mean(dim=1))
            decision_loss = nn.CrossEntropyLoss()(decisions, labels)
            decision_loss.backward()
            self.optimizer_decision.step()
            
        return {"pattern_loss": pattern_loss.item()}
    
    def predict_pattern(self, embedding: torch.Tensor):
        """Predict next pattern in sequence"""
        self.pattern_net.eval()
        with torch.no_grad():
            return self.pattern_net(embedding)
            
    def make_decision(self, embedding: torch.Tensor):
        """Make a decision based on current state"""
        self.decision_net.eval()
        with torch.no_grad():
            return self.decision_net(embedding.mean(dim=1))
            
    def store_experience(self, state, action, reward, next_state):
        """Store experience for learning"""
        self.memory_buffer.append((state, action, reward, next_state))
        
    def learn_from_experiences(self, batch_size: int = 32):
        """Learn from stored experiences"""
        if len(self.memory_buffer) < batch_size:
            return
            
        # Sample random batch
        batch = random.sample(self.memory_buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.stack([torch.tensor(s) for s in states])
        next_states = torch.stack([torch.tensor(s) for s in next_states])
        
        # Train on batch
        return self.train_on_batch(states)