submit50 slug
CS50
submit John-DAcquisto-3416/projects
import random

class NimAI:
    def get_q_value(self, state, action):
        """Return the Q-value for a given state-action pair."""
        return self.q.get((tuple(state), action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """Update Q-value using the Q-learning formula."""
        new_value = reward + future_rewards
        self.q[(tuple(state), action)] = old_q + self.alpha * (new_value - old_q)

    def best_future_reward(self, state):
        """Return the best possible reward for any available action in the given state."""
        possible_actions = [(i, j) for i, pile in enumerate(state) for j in range(1, pile + 1)]
        if not possible_actions:
            return 0
        return max(self.get_q_value(state, action) for action in possible_actions)

    def choose_action(self, state, epsilon=False):
        """Choose action using an epsilon-greedy approach or best action."""
        possible_actions = [(i, j) for i, pile in enumerate(state) for j in range(1, pile + 1)]
        if not possible_actions:
            return None

        if epsilon and random.random() < self.epsilon:
            return random.choice(possible_actions)

        return max(possible_actions, key=lambda action: self.get_q_value(state, action), default=None)
