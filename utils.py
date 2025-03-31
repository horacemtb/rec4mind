import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def dcg_at_k(relevance_scores, k):

    relevance_scores = np.array(relevance_scores[:k])
    return np.sum(relevance_scores / np.log2(np.arange(2, len(relevance_scores) + 2)))

def ndcg_at_k(relevance_scores, k):
    dcg = dcg_at_k(relevance_scores, k)

    # Ideal DCG (best possible ranking)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_relevance_scores, k)

    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(relevance_scores, k):
    relevance_scores = np.array(relevance_scores[:k])
    return np.sum(relevance_scores) / k

def recall_at_k(relevance_scores, k, total_relevant): # Requires total number of relevant items
    relevance_scores = np.array(relevance_scores[:k])
    return np.sum(relevance_scores) / total_relevant if total_relevant > 0 else 0.0

def is_hit(recommended, history):
    return int(any(item in history for item in recommended))

def calc_for_user(USER_ID, merged_behaviors, N, NUM_CYCLES, AT_K, env, model):

    user_hist = merged_behaviors[merged_behaviors['user_id'] == USER_ID].iloc[0, 1].split()[-N:]
    clicked_user_imressions = [i.split('-')[0] for i in merged_behaviors[merged_behaviors['user_id'] == USER_ID].iloc[0, 2].split() if int(i.split('-')[1])==1]

    obs = env.reset(user_hist)

    for i in range(NUM_CYCLES):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

    relevance_scores = env.rewards
    total_relevant = sum(relevance_scores)

    ndcg = ndcg_at_k(relevance_scores, AT_K)
    precision = precision_at_k(relevance_scores, AT_K)
    recall = recall_at_k(relevance_scores, AT_K, total_relevant)
    hit = is_hit(env.recommended, clicked_user_imressions)

    return ndcg, precision, recall, hit

def load_fasttext_embeddings(embeddings_path):
    """Loads fasttext embeddings from a pickle file and normalizes them."""
    with open(embeddings_path, 'rb') as f:
        item_emb = pickle.load(f)
    item_emb['fasttext_embedding'] = item_emb['fasttext_embedding'].apply(lambda x: x / np.linalg.norm(x))
    
    return dict(zip(item_emb['news_id'], item_emb['fasttext_embedding']))

def load_custom_embeddings(embeddings_path):
    """
    Loads precomputed custom embeddings from a pickle file."""
    with open(embeddings_path, 'rb') as f:
        item_emb = pickle.load(f)

    return item_emb

def load_bert_embeddings(embeddings_path):
    """
    Loads precomputed bert embeddings from a pickle file."""
    with open(embeddings_path, 'rb') as f:
        item_emb = pickle.load(f)
    item_emb['bert_embedding'] = item_emb['bert_embedding'].apply(lambda x: x / np.linalg.norm(x))
    
    return dict(zip(item_emb['news_id'], item_emb['bert_embedding']))

def load_entity_embeddings(embeddings_path):
    """
    Loads precomputed entity embeddings from a pickle file."""
    with open(embeddings_path, 'rb') as f:
        item_emb = pickle.load(f)

    return dict(zip(item_emb['news_id'], item_emb['entity_embedding']))

class GRUFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, emb_dim=64, hidden_dim=64):
        super().__init__(observation_space, features_dim=hidden_dim)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True) # input shape: (batch_size, L, emb_dim) where L is sequence length (num of items)

    def forward(self, observations):
        _, hidden = self.gru(observations) # (D*num_layers, batch, hidden_dim) where D == 2 if bidirectional=True otherwise 1
        return hidden.squeeze(0)  # Output shape: (batch, hidden_dim)
