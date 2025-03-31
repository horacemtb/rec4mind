import numpy as np
import gym
from gym import spaces
import pandas as pd


class RecommendationEnv(gym.Env):
    """
    A Gym environment for news recommendation using reinforcement learning.

    This environment simulates a user interacting with a recommender system.
    The user's state is represented as an averaged embedding of their history,
    and the action is an embedding vector representing the recommendation.
    The environment computes rewards based on the similarity between the state
    and the recommended articles' embeddings.

    Attributes:
        behaviors_df (pd.DataFrame): Dataset containing user behavior data.
        embeddings_dict (dict): Dictionary mapping article IDs to their embeddings.
        reward_processor (StateActionRewardProcessor): Module to compute rewards for state-action pairs.
        emb_dim (int): Dimensionality of the article embeddings.
        max_steps (int): Maximum number of steps per episode.
        gamma (float): Discount factor used in the reward calculation.
    """
    def __init__(self, behaviors_df, embeddings_dict, reward_processor, emb_dim=64, max_steps=10, gamma=0.9):
        super(RecommendationEnv, self).__init__()

        self.behaviors_df = behaviors_df
        self.embeddings_dict = embeddings_dict
        self.reward_processor = reward_processor
        self.emb_dim = emb_dim
        self.max_steps = max_steps
        self.gamma = gamma
        self.current_step = 0

        # Define state space (embedding of size emb_dim)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.emb_dim,), dtype=np.float32)

        # Define action space (embedding of size emb_dim)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.emb_dim,), dtype=np.float32)

        # Initialize user history and tracking variables
        self.user_history = []
        self.current_state = None
        self.done = False
        self.recommended = []
        self.rewards = []

    def reset(self, user_hist=None):
        """
        Resets the environment at the start of an episode.

        Args:
            user_hist (list, optional): A predefined user history for the episode.
        
        Returns:
            np.ndarray: Initial state embedding.
        """
        self.current_step = 0
        self.no_clicks_count = 0
        self.done = False

        if user_hist is None:
            # Pick a random user from the dataset
            user_row = self.behaviors_df.sample(1).iloc[0]
            history_articles = user_row["history"].split()
            history_articles = history_articles[-self.reward_processor.MAX_HISTORY:]
            self.user_history = [aid for aid in history_articles if aid in self.embeddings_dict]
        else:
            # Use provided user history
            self.user_history = [aid for aid in user_hist if aid in self.embeddings_dict]

        if not self.user_history:
            # If no valid history exists, set state to a zero vector
            self.current_state = np.zeros(self.emb_dim, dtype=np.float32)
        else:
            state_embeddings = np.array([self.embeddings_dict[aid] for aid in self.user_history])
            self.current_state = np.mean(state_embeddings, axis=0).astype(np.float32)

        self.recommended = []
        self.rewards = []

        return self.current_state

    def step(self, action):
        """
        Executes a step in the environment by recommending articles.

        Args:
            action (np.ndarray): The action vector representing the recommendation.

        Returns:
            tuple: (new state, reward, done flag, additional info)
        """
        self.current_step += 1

        # Get candidate articles (not in user history or already recommended)
        candidate_ids = [cid for cid in self.embeddings_dict.keys() if cid not in self.user_history and cid not in self.recommended]

        if not candidate_ids:
            # If no candidates remain, terminate the episode
            self.done = True
            return self.current_state, -1, self.done, {}

        candidate_embeddings = np.array([self.embeddings_dict[cid] for cid in candidate_ids])

        # Compute similarity scores and select top-K recommended items
        similarity_scores = np.dot(candidate_embeddings, action)
        top_indices = np.argsort(similarity_scores)[-self.reward_processor.K:]
        recommended_articles = [candidate_ids[i] for i in top_indices]
        self.recommended.extend(recommended_articles)

        # Compute reward using the simulator
        state_embedding = self.current_state
        weights = similarity_scores[top_indices] / np.sum(similarity_scores[top_indices])  # Normalize weights
        action_embedding = np.average([self.embeddings_dict[aid] for aid in recommended_articles], axis=0, weights=weights)
        rewards = self.reward_processor.get_reward(state_embedding, action_embedding)
        self.rewards.extend(rewards)

        reward = sum([self.gamma**k * reward for k, reward in enumerate(rewards)])

        # Update state if reward is positive
        if reward > 0:

            clicked_articles = [aid for aid, r in zip(recommended_articles, rewards) if r == 1]
            self.user_history.extend(clicked_articles)
            self.user_history = self.user_history[-self.reward_processor.MAX_HISTORY:]  # Keep max history length

            state_embeddings = np.array([self.embeddings_dict[aid] for aid in self.user_history])
            self.current_state = np.mean(state_embeddings, axis=0).astype(np.float32)

        # Define termination conditions
        if self.current_step >= self.max_steps:
            self.done = True

        if reward == 0:
            self.no_clicks_count += 1
        else:
            self.no_clicks_count = 0

        if self.no_clicks_count >= 3:
            self.done = True
            reward = -1

        return self.current_state, reward, self.done, {}


class RecurrentRecommendationEnv(gym.Env):
    """
    A recurrent Gym environment for news recommendation using reinforcement learning.

    This environment simulates a user interacting with a recommender system, where the user's state is represented as a 
    sequence of item embeddings. This sequential (recurrent) design is meant to be used with GRU-based feature extractors 
    that can capture temporal dependencies in the user's history.

    Attributes:
        behaviors_df (pd.DataFrame): Dataset containing user behaviors.
        embeddings_dict (dict): Dictionary mapping article IDs to their embeddings.
        reward_processor (StateActionRewardProcessor): Module for computing rewards based on state-action pairs.
        emb_dim (int): Dimensionality of the article embeddings.
        max_steps (int): Maximum number of steps per episode.
        gamma (float): Discount factor used in reward calculation.
    """
    def __init__(self, behaviors_df, embeddings_dict, reward_processor, emb_dim=64, max_steps=10, gamma=0.9):
        super(RecurrentRecommendationEnv, self).__init__()

        self.behaviors_df = behaviors_df
        self.embeddings_dict = embeddings_dict
        self.reward_processor = reward_processor
        self.emb_dim = emb_dim
        self.max_steps = max_steps
        self.gamma = gamma
        self.current_step = 0

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.reward_processor.MAX_HISTORY, self.emb_dim), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.reward_processor.K, self.emb_dim), dtype=np.float32)

        # Initialize user history and tracking variables
        self.user_history = []
        self.current_state = None
        self.done = False
        self.recommended = []
        self.rewards = []

    def reset(self, user_hist=None):

        self.current_step = 0
        self.no_clicks_count = 0
        self.done = False

        if user_hist is None:
            # Pick a random user from the dataset
            user_row = self.behaviors_df.sample(1).iloc[0]
            history_articles = user_row["history"].split()
            history_articles = history_articles[-self.reward_processor.MAX_HISTORY:]
            self.user_history = [aid for aid in history_articles if aid in self.embeddings_dict]
        else:
            # Use provided user history
            self.user_history = [aid for aid in user_hist if aid in self.embeddings_dict]

        if not self.user_history:
            # If no valid history exists, set state to a zero vector
            state_sequence = np.zeros((self.reward_processor.MAX_HISTORY, self.emb_dim), dtype=np.float32)
        else:
            state_embeddings = np.array([self.embeddings_dict[aid] for aid in self.user_history])
            if len(state_embeddings) < self.reward_processor.MAX_HISTORY:
                padding = np.zeros((self.reward_processor.MAX_HISTORY - len(state_embeddings), self.emb_dim))  # Pad with zeros
                state_embeddings = np.vstack([padding, state_embeddings])  # Stack to maintain fixed size
            else:
                state_embeddings = state_embeddings[-self.reward_processor.MAX_HISTORY:]  # Keep only last 6 items

            state_sequence = state_embeddings.astype(np.float32)

        self.recommended = []
        self.rewards = []

        self.current_state = state_sequence

        return self.current_state

    def step(self, action):
        """
        Executes a step in the environment by recommending articles.

        Args:
            action (np.ndarray): The action vector representing the recommendation.

        Returns:
            tuple: (new state, reward, done flag, additional info)
        """
        self.current_step += 1

        # Get candidate articles (not in user history or already recommended)
        candidate_ids = [cid for cid in self.embeddings_dict.keys() if cid not in self.user_history and cid not in self.recommended]

        if not candidate_ids:
            # If no candidates remain, terminate the episode
            self.done = True
            return self.current_state, -1, self.done, {}

        candidate_embeddings = np.array([self.embeddings_dict[cid] for cid in candidate_ids])

        # Compute similarity scores and select top-K recommended items
        similarity_scores = np.dot(candidate_embeddings, action.T)
        top_indices = np.argsort(similarity_scores, axis=0)[-1:]
        recommended_articles = [candidate_ids[i] for i in top_indices.flatten()]
        self.recommended.extend(recommended_articles)

        # Compute reward using the simulator
        state_embedding = self.current_state
        mask = np.all(state_embedding == 0, axis=1)
        filtered_embedding = state_embedding[~mask]
        state_embedding = np.mean(filtered_embedding, axis=0)
        action_embedding = np.mean([self.embeddings_dict[aid] for aid in recommended_articles], axis=0)
        rewards = self.reward_processor.get_reward(state_embedding, action_embedding)
        self.rewards.extend(rewards)

        reward = sum([self.gamma**k * reward for k, reward in enumerate(rewards)])

        # Update state if reward is positive
        if reward > 0:

            clicked_articles = [aid for aid, r in zip(recommended_articles, rewards) if r == 1]
            self.user_history.extend(clicked_articles)
            self.user_history = self.user_history[-self.reward_processor.MAX_HISTORY:]  # Keep max history length

            state_embeddings = np.array([self.embeddings_dict[aid] for aid in self.user_history])
            if len(state_embeddings) < self.reward_processor.MAX_HISTORY:
                padding = np.zeros((self.reward_processor.MAX_HISTORY - len(state_embeddings), self.emb_dim))
                state_embeddings = np.vstack([padding, state_embeddings])
            else:
                state_embeddings = state_embeddings[-self.reward_processor.MAX_HISTORY:]

            self.current_state = state_embeddings.astype(np.float32)

        # Define termination conditions
        if self.current_step >= self.max_steps:
            self.done = True

        if reward == 0:
            self.no_clicks_count += 1
        else:
            self.no_clicks_count = 0

        if self.no_clicks_count >= 3:
            self.done = True
            reward = -1

        return self.current_state, reward, self.done, {}