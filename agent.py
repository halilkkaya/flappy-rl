import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    """
    Deep Q-Network

    4 input → 64 hidden → 64 hidden → 2 output (zıpla/zıplama)
    """
    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class ReplayBuffer:
    """
    Experience Replay Buffer

    Geçmiş deneyimleri saklar ve rastgele batch'ler döndürür
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Deneyim ekle"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Rastgele batch örnekle"""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent

    Deep Q-Learning ile Flappy Bird oynamayı öğrenir
    """
    def __init__(
        self,
        state_size=4,
        action_size=2,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64
    ):
        """
        Args:
            state_size: State vektör boyutu (4)
            action_size: Action sayısı (2: zıpla/zıplama)
            learning_rate: Öğrenme hızı
            gamma: Discount factor (gelecek ödüllerin değeri)
            epsilon: Exploration oranı (başlangıç)
            epsilon_min: Minimum exploration oranı
            epsilon_decay: Epsilon azalma oranı
            buffer_size: Replay buffer boyutu
            batch_size: Eğitim batch boyutu
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # GPU varsa kullan
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Neural Network
        self.model = DQN(state_size, 64, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Replay Buffer
        self.memory = ReplayBuffer(buffer_size)

    def act(self, state, training=True):
        """
        State'e göre action seç

        Args:
            state: Mevcut durum
            training: Eğitim modunda mı? (epsilon-greedy için)

        Returns:
            int: Seçilen action (0 veya 1)
        """
        # Epsilon-greedy exploration (sadece eğitimde)
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploitation: En iyi action'ı seç
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Deneyimi hafızaya kaydet"""
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        """
        Replay buffer'dan örnekleyerek modeli eğit

        Returns:
            float: Loss değeri (None ise yeterli veri yok)
        """
        # Yeterli veri yoksa eğitme
        if len(self.memory) < self.batch_size:
            return None

        # Batch örnekle
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Numpy → Torch tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Mevcut Q değerleri
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Hedef Q değerleri (Bellman equation)
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss hesapla ve backprop
        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_epsilon(self):
        """Epsilon'u azalt (exploration → exploitation)"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        """Modeli kaydet"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Modeli yükle"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")


# Test için
if __name__ == "__main__":
    # Agent oluştur
    agent = DQNAgent()

    # Rastgele state
    state = np.random.rand(4)

    # Action seç
    action = agent.act(state)
    print(f"Selected action: {action}")

    # Deneyim kaydet
    next_state = np.random.rand(4)
    agent.remember(state, action, 1.0, next_state, False)

    print(f"Memory size: {len(agent.memory)}")
