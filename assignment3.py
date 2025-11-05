from typing import Dict, Tuple, List
import csv
import os

# COMP 3106 Fall 2025 Assignment 3
# Carleton University
# Student Implementation

# Group 133
# Rahul Rodrigues (101145082), Emily Amos (101311817), Abaan Norman (101305538)

class td_qlearning:

  alpha = 0.10
  gamma = 0.90

  def __init__(self, directory):
    # directory is the path to a directory containing trials through state space
    # Return nothing
    self.Q: Dict[Tuple[str, int], float] = {}
    self._states_seen: set[str] = set()

    # Load all trials (each trial is a list of (state, action))
    trials: List[List[Tuple[str, int]]] = []
    for fname in sorted(os.listdir(directory)):
      if not fname.lower().endswith('.csv'):
          continue
      path = os.path.join(directory, fname)
      trial: List[Tuple[str, int]] = []
      with open(path, 'r', newline='') as f:
          reader = csv.reader(f)
          for row in reader:
              if not row:
                  continue
              state = row[0].strip()
              # Some example files may have a dash for terminal line's action.
              action_field = row[1].strip() if len(row) > 1 else ''
              if action_field == '-' or action_field == '':
                  # For terminal states, store an action of 0 (never selected).
                  action = 0
              else:
                  action = int(action_field)
              trial.append((state, action))
              self._states_seen.add(state)
      if trial:
          trials.append(trial)

    # Initialize Q(s,a) = r(s) for all (s,a) that appear/are feasible
    # We'll lazily initialize unseen (s,a) pairs during updates as needed.
    for trial in trials:
      for (s, a) in trial:
        for act in self._available_actions(s) if not self._is_terminal(s) else [0]:
          key = (s, act)
          if key not in self.Q:
              self.Q[key] = self._reward(s)

    # Iterate to convergence over all trials
    tol = 1e-8
    max_epochs = 20000  # Safe upper bound; will usually converge far earlier.
    for _ in range(max_epochs):
      delta = 0.0
      for trial in trials:
        # Each step uses (s_t, a_t) and next state s_{t+1}
        for idx in range(len(trial) - 1):
          s_t, a_t = trial[idx]
          s_tp1, _ = trial[idx + 1]

          if self._is_terminal(s_t):
            # No update from a terminal state.
            continue

          # Ensure Q entries exist
          if (s_t, a_t) not in self.Q:
            self.Q[(s_t, a_t)] = self._reward(s_t)

          # Target: r(s') + gamma * max_a' Q(s', a')
          r_tp1 = self._reward(s_tp1)
          if self._is_terminal(s_tp1):
            max_q_tp1 = 0.0  # No future actions beyond terminal
          else:
            max_q_tp1 = self._max_q_over_actions(s_tp1)

          target = self.gamma * r_tp1 + self.gamma * max_q_tp1
          old = self.Q[(s_t, a_t)]
          new = old + self.alpha * (target - old)
          self.Q[(s_t, a_t)] = new
          delta = max(delta, abs(new - old))

        # Optional: we could also ensure terminal state's Q(s,0)=r(s) stays fixed
        s_last, a_last = trial[-1]
        if self._is_terminal(s_last):
          key = (s_last, 0)
          val = self._reward(s_last)
          if key not in self.Q or self.Q[key] != val:
            self.Q[key] = val

      if delta < tol:
        break

  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is an integer representation of an action
    # Return the q-value for the state-action pair
    key = (state, int(action))
    if key not in self.Q:
      # Initialize on demand as per spec: Q(s,a) = r(s)
      self.Q[key] = self._reward(state)
    # Return the q-value for the state-action pair
    return self.Q[key]

  def policy(self, state):
    # state is a string representation of a state
    # Return the optimal action (as an integer) under the learned policy
    if self._is_terminal(state):
      # No legal action; return 0 to indicate terminal (won't be scored in tests)
      return 0
    best_a = None
    best_q = float('-inf')
    for a in self._available_actions(state):
      q = self.Q.get((state, a), self._reward(state))
      # tie-break prefers largest action
      if q > best_q or (q == best_q and (best_a is None or a > best_a)):
        best_q = q
        best_a = a
    # Return the optimal action (as an integer) under the learned policy
    return best_a if best_a is not None else 1

  # -------------- Helpers --------------
  @staticmethod
  def _parse_state(state: str) -> Tuple[int, int, int, str]:
    # Accept either ASCII '-' or Unicode '−' for non-terminal.
    parts = state.strip().split('/')
    if len(parts) != 4:
      raise ValueError(f"Malformed state string: {state}")
    cbag = int(parts[0])
    cagent = int(parts[1])
    copp = int(parts[2])
    winner = parts[3]
    # Normalize minus sign variants
    if winner == '−':
      winner = '-'
    return cbag, cagent, copp, winner

  @classmethod
  def _reward(cls, state: str) -> float:
    _, cagent, _, winner = cls._parse_state(state)
    if winner == 'A':
      return float(cagent)
    if winner == 'O':
      return float(-cagent)
    return 0.0

  @classmethod
  def _is_terminal(cls, state: str) -> bool:
    _, _, _, winner = cls._parse_state(state)
    return winner in ('A', 'O')

  @classmethod
  def _available_actions(cls, state: str) -> List[int]:
    cbag, _, _, winner = cls._parse_state(state)
    if winner in ('A', 'O'):
      return []
    max_take = min(3, max(0, cbag))
    return [a for a in (1, 2, 3) if a <= max_take and a > 0]

  def _max_q_over_actions(self, state: str) -> float:
    best = float('-inf')
    for a in self._available_actions(state):
      q = self.Q.get((state, a), self._reward(state))
      if q > best:
        best = q
    return best if best != float('-inf') else 0.0

