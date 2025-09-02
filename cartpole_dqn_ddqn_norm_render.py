# cartpole_dqn_ddqn_norm_render.py
# DQN + Double-DQN + Running state normalization + robust checkpoints + optional live rendering

import argparse
import math
import os
import random
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Utilities

def make_device():
   if torch.cuda.is_available():
      return torch.device("cuda")
   if torch.backends.mps.is_available():
      return torch.device("mps")
   return torch.device("cpu")


def set_seed(seed: int):
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)



# Running Normalizer (Welford) #

class RunningNorm:
   """
    Online mean/variance tracker for observation normalization.
    Stores primitive Python lists/floats in checkpoints (pickle-safe).
    """
   def __init__(self, shape, eps=1e-8):
      self.eps = eps
      self.shape = shape
      self.reset()

   def reset(self):
      self.mean = np.zeros(self.shape, dtype=np.float32)
      self.var = np.ones(self.shape, dtype=np.float32)
      self.count = 1e-4  # avoid div-by-zero for very early steps

   def update(self, x: np.ndarray):
      # x shape: (..., obs_dim)
      x = np.asarray(x, dtype=np.float32)
      batch = x.shape[0] if x.ndim > 1 else 1
      x_mean = x.mean(axis=0)
      x_var = x.var(axis=0)
      x_count = batch
   
      delta = x_mean - self.mean
      tot = self.count + x_count
   
      new_mean = self.mean + delta * (x_count / tot)
      m_a = self.var * self.count
      m_b = x_var * x_count
      M2 = m_a + m_b + np.square(delta) * (self.count * x_count / tot)
      new_var = M2 / tot
   
      self.mean = new_mean
      self.var = np.maximum(new_var, 1e-6)
      self.count = tot

   def normalize(self, x: np.ndarray):
      return (x - self.mean) / np.sqrt(self.var + self.eps)


# Replay Buffer #
Transition = namedtuple("Transition", ["s", "a", "r", "s2", "d"])

class ReplayBuffer:
   def __init__(self, obs_dim, size=100_000):
      self.size = size
      self.obs = np.zeros((size, obs_dim), dtype=np.float32)
      self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
      self.acts = np.zeros((size,), dtype=np.int64)
      self.rews = np.zeros((size,), dtype=np.float32)
      self.dones = np.zeros((size,), dtype=np.float32)
      self.ptr = 0
      self.count = 0

   def push(self, s, a, r, s2, d):
      idx = self.ptr % self.size
      self.obs[idx] = s
      self.acts[idx] = a
      self.rews[idx] = r
      self.next_obs[idx] = s2
      self.dones[idx] = float(d)
      self.ptr += 1
      self.count = min(self.count + 1, self.size)

   def __len__(self):
      return self.count

   def sample(self, batch_size=128):
      idxs = np.random.randint(0, self.count, size=batch_size)
      b_s = self.obs[idxs]
      b_a = self.acts[idxs]
      b_r = self.rews[idxs]
      b_s2 = self.next_obs[idxs]
      b_d = self.dones[idxs]
      return b_s, b_a, b_r, b_s2, b_d


# Q-Network
class QNet(nn.Module):
   def __init__(self, obs_dim, act_dim, hidden=128):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(obs_dim, hidden),
         nn.ReLU(),
         nn.Linear(hidden, hidden),
         nn.ReLU(),
         nn.Linear(hidden, act_dim),
         )
   
      # Nice default init for stability
      for m in self.modules():
         if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
               fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
               bound = 1 / math.sqrt(fan_in)
               nn.init.uniform_(m.bias, -bound, bound)

   def forward(self, x):
      return self.net(x)


# Epsilon schedule

class EpsGreedy:
   def __init__(self, eps_start=1.0, eps_end=0.05, decay_steps=20000):
      self.start = eps_start
      self.end = eps_end
      self.decay = decay_steps
      self.t = 0

   def value(self):
      frac = min(1.0, self.t / self.decay)
      return self.start + (self.end - self.start) * frac

   def step(self):
      self.t += 1



# Select action

def select_action(qnet, obs_normed, eps, act_dim, device):
   if random.random() < eps:
      return random.randrange(act_dim)
   with torch.no_grad():
      q = qnet(torch.from_numpy(obs_normed).float().unsqueeze(0).to(device))
      return int(q.argmax(dim=1).item())



# Training (Double-DQN)
def train(args):
   device = make_device()
   set_seed(args.seed)

   # Training env (no render by default for speed)
   env = gym.make("CartPole-v1")
   obs_dim = env.observation_space.shape[0]
   act_dim = env.action_space.n

   online = QNet(obs_dim, act_dim, hidden=args.hidden).to(device)
   target = QNet(obs_dim, act_dim, hidden=args.hidden).to(device)
   target.load_state_dict(online.state_dict())

   opt = optim.Adam(online.parameters(), lr=args.lr)
   loss_fn = nn.SmoothL1Loss()
   rb = ReplayBuffer(obs_dim, size=args.replay_size)
   norm = RunningNorm(obs_dim)

   eps_sched = EpsGreedy(args.eps_start, args.eps_end, args.eps_decay)

   ep_ret = 0.0
   ep_len = 0
   returns = deque(maxlen=20)
   best_avg20 = -float("inf")

   state, _ = env.reset(seed=args.seed)
   norm.update(state)  # bootstrap stats

   step = 0
   episode = 0

   # preview training visually every N episodes using a separate env
   def preview_episode():
      if args.render_train_freq <= 0:
         return
      if episode % args.render_train_freq != 0:
         return
      eval_env = gym.make("CartPole-v1", render_mode="human")
      s, _ = eval_env.reset()
      done = False
      # Use near-greedy policy for preview
      while not done:
         s_norm = norm.normalize(s)
         a = select_action(online, s_norm, eps=0.01, act_dim=act_dim, device=device)
         s2, r, terminated, truncated, _ = eval_env.step(a)
         done = terminated or truncated
         s = s2
      eval_env.close()

   while step < args.total_steps:
      s_norm = norm.normalize(state)
      eps = max(args.eps_min, eps_sched.value())
      action = select_action(online, s_norm, eps, act_dim, device)
   
      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
      ep_ret += reward
      ep_len += 1
   
      # Update normalizer & store transition
      norm.update(next_state)
      rb.push(state.astype(np.float32), action, reward, next_state.astype(np.float32), done)
   
      state = next_state
      step += 1
      eps_sched.step()
   
      # Learn
      if len(rb) >= args.learn_start and step % args.learn_every == 0:
         for _ in range(args.gradient_updates):
            b_s, b_a, b_r, b_s2, b_d = rb.sample(args.batch_size)
         
            b_s_n = norm.normalize(b_s)
            b_s2_n = norm.normalize(b_s2)
         
            s_t = torch.from_numpy(b_s_n).float().to(device)
            a_t = torch.from_numpy(b_a).long().to(device).unsqueeze(1)
            r_t = torch.from_numpy(b_r).float().to(device).unsqueeze(1)
            s2_t = torch.from_numpy(b_s2_n).float().to(device)
            d_t = torch.from_numpy(b_d).float().to(device).unsqueeze(1)
         
            # Q(s,a)
            q_sa = online(s_t).gather(1, a_t)
         
            # Double-DQN target
            with torch.no_grad():
               next_actions = online(s2_t).argmax(dim=1, keepdim=True)
               q_target_next = target(s2_t).gather(1, next_actions)
               y = r_t + args.gamma * (1.0 - d_t) * q_target_next
         
            loss = loss_fn(q_sa, y)
         
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
            opt.step()
   
      # Target update
      if step % args.target_update_every == 0:
         target.load_state_dict(online.state_dict())
   
      # End of episode
      if done:
         returns.append(ep_ret)
         episode += 1
      
         avg20 = np.mean(returns) if returns else ep_ret
      
         if episode % args.log_every == 0:
            print(
               f"Step {step:7d} | EpScore {ep_ret:4.1f} | Avg20 {avg20:5.2f} | "
               f"eps {eps:0.3f} | RB {len(rb)}"
               )
      
         # Save best (robust, pickle-safe checkpoint)
         if avg20 > best_avg20 and len(returns) == returns.maxlen:
            best_avg20 = avg20
            ckpt = {
               "model": online.state_dict(),
               "norm": {
                   "mean": norm.mean.tolist(),
                   "var": norm.var.tolist(),
                   "count": float(norm.count),
               },
               }
            torch.save(ckpt, args.best_path)
            # optional: also save a last checkpoint
            torch.save(ckpt, args.last_path)
      
         # Optional live preview of current policy
         preview_episode()
      
         # Reset episode
         ep_ret = 0.0
         ep_len = 0
         state, _ = env.reset()

   env.close()

   print(f"\nLoaded best (Avg20≈{best_avg20:.1f}).", flush=True)
   evaluate(args, model_path=args.best_path, n_episodes=10, render=args.render_eval)



# Robust evaluate (safe load)
   
def _safe_load_checkpoint(path, map_location="cpu"):
   """
    Robust loader for PyTorch 2.6+. Tries weights_only=True first,
    then falls back to weights_only=False for backward-compat.
    """
   try:
      # Default in PT 2.6 is weights_only=True; keep explicit for clarity
      return torch.load(path, map_location=map_location)  # weights_only=True by default
   except Exception:
      # If checkpoint contains older pickled numpy objects, fall back
      return torch.load(path, map_location=map_location, weights_only=False)


def evaluate(args, model_path, n_episodes=10, render=False):
   device = make_device()

   # Build a fresh online net to load weights into
   env = gym.make("CartPole-v1", render_mode="human" if render else None)
   obs_dim = env.observation_space.shape[0]
   act_dim = env.action_space.n

   online = QNet(obs_dim, act_dim, hidden=args.hidden).to(device)
   norm = RunningNorm(obs_dim)

   # Load checkpoint (new or old format)
   ckpt = _safe_load_checkpoint(model_path, map_location=device)
   online.load_state_dict(ckpt["model"])

   if "norm" in ckpt:  # new format
      norm.mean = np.array(ckpt["norm"]["mean"], dtype=np.float32)
      norm.var = np.array(ckpt["norm"]["var"], dtype=np.float32)
      norm.count = float(ckpt["norm"]["count"])
   else:  # backward-compat (older code)
      # old flat keys: "norm_mean", "norm_var", "norm_count"
      norm.mean = np.array(ckpt["norm_mean"], dtype=np.float32)
      norm.var = np.array(ckpt["norm_var"], dtype=np.float32)
      norm.count = float(ckpt["norm_count"])

   online.eval()
   scores = []
   for _ in range(n_episodes):
      s, _ = env.reset(seed=random.randint(0, 999_999))
      done = False
      ep = 0.0
      while not done:
         s_n = norm.normalize(s)
         with torch.no_grad():
            q = online(torch.from_numpy(s_n).float().unsqueeze(0).to(device))
            a = int(q.argmax(dim=1).item())
         s, r, terminated, truncated, _ = env.step(a)
         done = terminated or truncated
         ep += r
         if render:
            env.render()
      scores.append(ep)

   env.close()
   print(f"Eval avg return over {n_episodes} eps: {np.mean(scores):.1f}")



# Main / CLI

def parse_args():
   p = argparse.ArgumentParser()
   # Training
   p.add_argument("--total-steps", type=int, default=300_000)
   p.add_argument("--replay-size", type=int, default=100_000)
   p.add_argument("--batch-size", type=int, default=128)
   p.add_argument("--learn-start", type=int, default=1_000)
   p.add_argument("--learn-every", type=int, default=4)
   p.add_argument("--gradient-updates", type=int, default=1)
   p.add_argument("--target-update-every", type=int, default=1_000)
   p.add_argument("--gamma", type=float, default=0.99)
   p.add_argument("--lr", type=float, default=1e-3)
   p.add_argument("--hidden", type=int, default=128)
   p.add_argument("--seed", type=int, default=42)

   # Epsilon
   p.add_argument("--eps-start", type=float, default=1.0)
   p.add_argument("--eps-end", type=float, default=0.05)
   p.add_argument("--eps-decay", type=int, default=20_000)
   p.add_argument("--eps-min", type=float, default=0.01)

   # Rendering
   p.add_argument("--render-eval", action="store_true", help="Render evaluation episodes.")
   p.add_argument("--render-train-freq", type=int, default=0,
                 help="Every N episodes, preview one live episode during training (0=off).")

   # Logging / files
   p.add_argument("--log-every", type=int, default=500)
   p.add_argument("--best-path", type=str, default="cartpole_dqn_best.pt")
   p.add_argument("--last-path", type=str, default="cartpole_dqn_last.pt")

   # Modes
   p.add_argument("--eval-only", action="store_true",
                 help="Skip training; just load checkpoint and run evaluation.")
   return p.parse_args()


if __name__ == "__main__":
   args = parse_args()

   if args.eval_only:
      if not os.path.exists(args.best_path):
         raise FileNotFoundError(f"Checkpoint not found: {args.best_path}")
      evaluate(args, model_path=args.best_path, n_episodes=10, render=args.render_eval)
   else:
      train(args)
