"""
Four DRL agents for the comparative study:
  1. DQN          – vanilla Deep Q-Network
  2. Double DQN   – decouples action selection from evaluation
  3. Dueling DQN  – separate value / advantage streams
  4. A2C          – Advantage Actor-Critic (on-policy)

All value-based agents share a common replay buffer and ε-greedy
exploration; A2C collects trajectories and updates after each episode.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

# ═══════════════════════════════════════════════════════════════════
#  Replay Buffer (shared by DQN variants)
# ═══════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ═══════════════════════════════════════════════════════════════════
#  Network architectures
# ═══════════════════════════════════════════════════════════════════

class QNet(nn.Module):
    """Standard fully-connected Q-network."""
    def __init__(self, state_dim, action_dim, hidden=(256, 128)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DuelingQNet(nn.Module):
    """
    Dueling architecture (Wang et al. 2016).
    Splits into Value V(s) and Advantage A(s,a) streams.
    Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]
    """
    def __init__(self, state_dim, action_dim, hidden=(256, 128)):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden[0]),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], action_dim),
        )

    def forward(self, x):
        feat = self.feature(x)
        v = self.value(feat)                         # (B, 1)
        a = self.advantage(feat)                     # (B, A)
        return v + a - a.mean(dim=1, keepdim=True)   # (B, A)


class ActorCriticNet(nn.Module):
    """Shared-trunk actor-critic for A2C."""
    def __init__(self, state_dim, action_dim, hidden=(256, 128)):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden[0]), nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def act(self, x):
        logits, val = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), val.squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
#  1. Standard DQN Agent
# ═══════════════════════════════════════════════════════════════════

class DQNAgent:
    name = "DQN"

    def __init__(self, state_dim, action_dim, *,
                 lr=1e-4, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.998,
                 buffer_size=50_000, batch_size=64,
                 target_update=1000, hidden=(256, 128)):
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self._steps = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.q = self._make_net(state_dim, action_dim, hidden).to(self.device)
        self.q_target = self._make_net(state_dim, action_dim, hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buf = ReplayBuffer(buffer_size)
        self.losses: list[float] = []

    @staticmethod
    def _make_net(s, a, h):
        return QNet(s, a, h)

    # ─── action ──────────────────────────────────────────────────
    def select_action(self, state, training=True):
        if training and random.random() < self.eps:
            return random.randrange(self.action_dim)
        t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.q(t).argmax(1).item()

    # ─── store ───────────────────────────────────────────────────
    def store(self, s, a, r, s2, d):
        self.buf.push(s, a, r, s2, d)

    # ─── learn ───────────────────────────────────────────────────
    def update(self):
        if len(self.buf) < self.batch_size:
            return
        s, a, r, s2, d = self.buf.sample(self.batch_size)
        s  = torch.as_tensor(s,  device=self.device)
        a  = torch.as_tensor(a,  device=self.device)
        r  = torch.as_tensor(r,  device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        d  = torch.as_tensor(d,  device=self.device)

        q_val = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self._target_values(s2)
            target = r + self.gamma * q_next * (1 - d)

        loss = nn.SmoothL1Loss()(q_val, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.losses.append(loss.item())

        self._steps += 1
        if self._steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def _target_values(self, s2):
        """Standard DQN: target net picks max Q."""
        return self.q_target(s2).max(1)[0]

    # ─── save / load ─────────────────────────────────────────────
    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'q': self.q.state_dict(), 'qt': self.q_target.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.q.load_state_dict(ck['q'])
        self.q_target.load_state_dict(ck['qt'])


# ═══════════════════════════════════════════════════════════════════
#  2. Double DQN Agent
# ═══════════════════════════════════════════════════════════════════

class DoubleDQNAgent(DQNAgent):
    """
    Double DQN (van Hasselt et al. 2016).
    Online network selects the action; target network evaluates it.
    """
    name = "Double DQN"

    def _target_values(self, s2):
        best_actions = self.q(s2).argmax(1)                      # online selects
        return self.q_target(s2).gather(1, best_actions.unsqueeze(1)).squeeze(1)  # target evaluates


# ═══════════════════════════════════════════════════════════════════
#  3. Dueling DQN Agent
# ═══════════════════════════════════════════════════════════════════

class DuelingDQNAgent(DQNAgent):
    """
    Dueling DQN (Wang et al. 2016).
    Uses DuelingQNet instead of QNet; otherwise same as standard DQN.
    """
    name = "Dueling DQN"

    @staticmethod
    def _make_net(s, a, h):
        return DuelingQNet(s, a, h)


# ═══════════════════════════════════════════════════════════════════
#  4. A2C Agent
# ═══════════════════════════════════════════════════════════════════

class A2CAgent:
    """
    Advantage Actor-Critic (synchronous, single env).
    Collects full episode then does one gradient step.
    """
    name = "A2C"

    def __init__(self, state_dim, action_dim, *,
                 lr=3e-4, gamma=0.99, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=0.5,
                 hidden=(256, 128)):
        self.gamma = gamma
        self.ent_c = entropy_coef
        self.val_c = value_coef
        self.max_grad = max_grad_norm
        self.action_dim = action_dim

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = ActorCriticNet(state_dim, action_dim, hidden).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.losses: list[float] = []

        # trajectory buffers (cleared each update)
        self._states, self._actions, self._rewards = [], [], []
        self._log_probs, self._values, self._dones = [], [], []

    def select_action(self, state, training=True):
        t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if training:
            a, lp, v = self.net.act(t)
            self._log_probs.append(lp)
            self._values.append(v)
            return a.item()
        with torch.no_grad():
            logits, _ = self.net(t)
            return logits.argmax(1).item()

    def store(self, s, a, r, s2, d):
        self._states.append(s)
        self._actions.append(a)
        self._rewards.append(r)
        self._dones.append(d)

    def update(self):
        if not self._rewards:
            return
        # compute discounted returns
        R, returns = 0.0, []
        for r, d in zip(reversed(self._rewards), reversed(self._dones)):
            R = r + self.gamma * R * (1.0 - d)
            returns.insert(0, R)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        if ret.std() > 0:
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        lp = torch.stack(self._log_probs).squeeze()
        val = torch.stack(self._values).squeeze()
        adv = ret - val.detach()

        actor_loss = -(lp * adv).mean()
        critic_loss = nn.MSELoss()(val, ret)

        # entropy bonus
        st = torch.as_tensor(np.array(self._states), dtype=torch.float32, device=self.device)
        logits, _ = self.net(st)
        p = torch.softmax(logits, -1)
        entropy = -(p * (p + 1e-8).log()).sum(-1).mean()

        loss = actor_loss + self.val_c * critic_loss - self.ent_c * entropy
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
        self.opt.step()
        self.losses.append(loss.item())

        # clear buffers
        self._states.clear(); self._actions.clear(); self._rewards.clear()
        self._log_probs.clear(); self._values.clear(); self._dones.clear()

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'net': self.net.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ck['net'])
