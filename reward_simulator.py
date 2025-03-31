import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


class StateActionRewardProcessor:
    def __init__(self, behaviors_df, embeddings_path_fasttext=None, embeddings_path_custom=None, embeddings_path_bert=None, embeddings_path_ent=None, 
                K=4, MIN_HISTORY=3, MAX_HISTORY=12, fraction=1, alpha=0.2):
        """
        Initializes the StateActionRewardProcessor by:
        - Preprocessing the behaviors dataset into (state, action, reward) pairs.
        - Dynamically loading and normalizing embeddings from different sources (FastText, custom, BERT, Entity).
        - Concatenating available embeddings for each item.
        - Precomputing mean embeddings for states and actions.

        Args:
            behaviors_df (pd.DataFrame): DataFrame with user behaviors (history, impressions).
            embeddings_path_fasttext (str, optional): Path to precomputed FastText embeddings.
            embeddings_path_custom (str, optional): Path to precomputed custom embeddings.
            embeddings_path_bert (str, optional): Path to precomputed BERT embeddings.
            embeddings_path_ent (str, optional): Path to precomputed entity embeddings.
            K (int): Number of recommended items per action.
            MIN_HISTORY (int): Minimum required history length for a valid state.
            MAX_HISTORY (int): Maximum history length for a state.
            fraction (float): Fraction of the dataset to use for processing (if embeddings don't fit into RAM).
            alpha (float): Weight for combining state and action similarity in reward retrieval.
        """
        self.K = K
        self.MIN_HISTORY = MIN_HISTORY
        self.MAX_HISTORY = MAX_HISTORY
        self.fraction = fraction
        self.alpha = alpha

        # Load embeddings
        self.embeddings_dict = {}

        if embeddings_path_fasttext:
            self.embeddings_dict_fasttext = self._load_fasttext_embeddings(embeddings_path_fasttext)
        else:
            self.embeddings_dict_fasttext = {}

        if embeddings_path_custom:
            self.embeddings_dict_custom = self._load_custom_embeddings(embeddings_path_custom)
        else:
            self.embeddings_dict_custom = {}

        if embeddings_path_bert:
            self.embedding_dict_bert = self._load_bert_embeddings(embeddings_path_bert)
        else:
            self.embedding_dict_bert = {}

        if embeddings_path_ent:
            self.embeddings_dict_ent = self._load_entity_embeddings(embeddings_path_ent)
        else:
            self.embeddings_dict_ent = {}

        # Get all possible keys (news_ids) from available embeddings
        all_keys = set(self.embedding_dict_bert.keys()) | set(self.embeddings_dict_ent.keys()) | set(self.embeddings_dict_custom.keys()) | set(self.embeddings_dict_fasttext.keys())

        for key in all_keys:

            emb_list = []

            if key in self.embeddings_dict_fasttext:
                emb_list.append(self.embeddings_dict_fasttext[key])
            if key in self.embeddings_dict_custom:
                emb_list.append(self.embeddings_dict_custom[key])
            if key in self.embedding_dict_bert:
                emb_list.append(self.embedding_dict_bert[key])
            if key in self.embeddings_dict_ent:
                emb_list.append(self.embeddings_dict_ent[key])

            if emb_list:
                self.embeddings_dict[key] = np.concatenate(emb_list)

        # Free memory
        del self.embeddings_dict_fasttext
        del self.embeddings_dict_custom
        del self.embedding_dict_bert
        del self.embeddings_dict_ent

        # Preprocess behaviors dataset
        self.state_action_reward_df = self._process_behaviors(behaviors_df if self.fraction == 1 else behaviors_df.sample(frac=self.fraction, random_state=9))

        # Compute average state and action embeddings
        self.averaged_state_action_embeddings_df = self._compute_average_embeddings()

        # Store precomputed embeddings for fast similarity lookup
        self.state_embeddings = np.array(self.averaged_state_action_embeddings_df['state_embedding'].tolist())
        self.action_embeddings = np.array(self.averaged_state_action_embeddings_df['action_embedding'].tolist())

    def _load_fasttext_embeddings(self, embeddings_path):
        """
        Loads precomputed FastText embeddings from a pickle file and normalizes them.

        Args:
            embeddings_path (str): Path to the FastText embeddings file.

        Returns:
            dict: A dictionary mapping item IDs to normalized FastText embeddings.
        """
        with open(embeddings_path, 'rb') as f:
            item_emb = pickle.load(f)

        item_emb['fasttext_embedding'] = item_emb['fasttext_embedding'].apply(lambda x: x / np.linalg.norm(x))
        return dict(zip(item_emb['news_id'], item_emb['fasttext_embedding']))

    def _load_custom_embeddings(self, embeddings_path):
        """
        Loads precomputed custom embeddings from a pickle file.

        Args:
            embeddings_path (str): Path to the custom embeddings file.

        Returns:
            dict: A dictionary mapping item IDs to custom embeddings.
        """

        with open(embeddings_path, 'rb') as f:
            item_emb = pickle.load(f)

        return item_emb

    def _load_bert_embeddings(self, embeddings_path):
        """
        Loads precomputed BERT embeddings from a pickle file and normalizes them.

        Args:
            embeddings_path (str): Path to the BERT embeddings file.

        Returns:
            dict: A dictionary mapping item IDs to normalized BERT embeddings.
        """
        with open(embeddings_path, 'rb') as f:
            item_emb = pickle.load(f)

        item_emb['bert_embedding'] = item_emb['bert_embedding'].apply(lambda x: x / np.linalg.norm(x))
        return dict(zip(item_emb['news_id'], item_emb['bert_embedding']))

    def _load_entity_embeddings(self, embeddings_path):
        """
        Loads precomputed entity embeddings from a pickle file.

        Args:
            embeddings_path (str): Path to the entity embeddings file.

        Returns:
            dict: A dictionary mapping item IDs to entity embeddings.
        """
        with open(embeddings_path, 'rb') as f:
            item_emb = pickle.load(f)

        return dict(zip(item_emb['news_id'], item_emb['entity_embedding']))

    def _process_behaviors(self, behaviors_df):
        """
        Processes user behaviors to extract (state, action, reward) pairs.

        Args:
            behaviors_df (pd.DataFrame): DataFrame with columns ['user_id', 'history', 'impressions'].

        Returns:
            pd.DataFrame: DataFrame with columns ['state', 'action', 'reward'].
        """
        state_action_reward_list = []

        for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), desc="Processing sessions"):

            user_id, history_str, impressions_str = row["user_id"], row["history"], row["impressions"]
            history = history_str.split()
            impressions = [imp.split('-') for imp in impressions_str.split()]
            impressions = [(article_id, int(click)) for article_id, click in impressions]

            if len(history) < self.MIN_HISTORY or len(impressions) < self.K:
                continue

            history = history[-self.MAX_HISTORY:]

            for i in range(0, len(impressions) - self.K + 1, self.K):
                action_window = impressions[i:i+self.K]
                action = [article_id for article_id, _ in action_window]
                reward = [click for _, click in action_window]

                state_action_reward_list.append({
                    "state": history.copy(),
                    "action": action,
                    "reward": reward
                })

                # Update history with clicked articles while maintaining size limit
                clicked_articles = [article_id for article_id, click in action_window if click == 1]
                history.extend(clicked_articles)
                history = history[-self.MAX_HISTORY:]

        return pd.DataFrame(state_action_reward_list)

    def _compute_average_embeddings(self):
        """
        Computes average embeddings for states and actions.

        Returns:
            pd.DataFrame: DataFrame with columns ['state_embedding', 'action_embedding', 'reward'].
        """
        embeddings = []

        for _, row in tqdm(self.state_action_reward_df.iterrows(), desc="Computing state-action embeddings"):

            state_embeddings = np.array([self.embeddings_dict[i] for i in row['state'] if i in self.embeddings_dict])
            action_embeddings = np.array([self.embeddings_dict[i] for i in row['action'] if i in self.embeddings_dict])

            avg_state_embedding = np.mean(state_embeddings, axis=0)
            avg_action_embedding = np.mean(action_embeddings, axis=0)

            embeddings.append({
                'state_embedding': avg_state_embedding,
                'action_embedding': avg_action_embedding,
                'reward': row['reward']
            })

        return pd.DataFrame(embeddings)

    def get_reward(self, state_embeddings, action_embeddings):
        """
        Finds the closest matching state-action pair and returns its reward.

        Args:
            state_embeddings (numpy.ndarray): Mean embedding of the state articles.
            action_embeddings (numpy.ndarray): Mean embedding of the action articles.

        Returns:
            list: Reward vector for the most similar state-action pair.
        """
        if state_embeddings is None or action_embeddings is None:
            return None

        cosine_state = np.dot(state_embeddings, self.state_embeddings.T) / (np.linalg.norm(state_embeddings) * np.linalg.norm(self.state_embeddings, axis=1))
        cosine_action = np.dot(action_embeddings, self.action_embeddings.T) / (np.linalg.norm(action_embeddings) * np.linalg.norm(self.action_embeddings, axis=1))

        combined_similarity = self.alpha * cosine_state + (1 - self.alpha) * cosine_action
        max_index = np.argmax(combined_similarity)

        return self.averaged_state_action_embeddings_df.iloc[max_index]['reward']
