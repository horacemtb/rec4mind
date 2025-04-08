# rec4mind
Reinforcement learning meets NLP: A news recommendation system trained on the MIND-small dataset

## Sources

This section provides links to the original article I derived the idea from and the data used in the project.

__Article__: https://arxiv.org/abs/1801.00209

__Data__: https://msnews.github.io/

__First Look__: https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md

## Commentary

The __research__ folder contains the following notebooks:

- __first_look.ipynb__ – explores the MIND dataset and derives insights through visualizations and analytics.
- __item_embeddings.ipynb__ – demonstrates how to obtain item representations (e.g., fastText, Sentence-BERT, and MIND entity embeddings) that serve as user states and actions in the reward simulator and environment.
- __custom_embeddings.ipynb__ – shows how to train custom embeddings using contrastive learning with a neural network architecture.
- __reward_simulator.ipynb__ – builds a reward simulator and explains its logic, including data preprocessing and the reward function.

The __embeddings__ folder contains precomputed embeddings for the train and test sets.

The __processed_data__ folder includes preprocessed user sessions, each consisting of browsing history, recommended items, and historical ground truth labels. These are used by the reward simulator and describe the environment state.

The __results__ folder contains reward and episode length logs recorded during the training of five RL algorithms: DDPG, A2C, PPO, SAC, and TD3. The A2C model was trained using different types of embeddings, such as baseline fastText, custom embeddings, and a combination of both. This data is visualized through learning curves found in the __images__ folder.

The production codebase consists of the following scripts:

- __reward_simulator.py__ – contains the reward simulator, which can be imported and used to provide feedback during model training.
- __utils.py__ – includes helper functions for calculating metrics, evaluating performance on train/test sets, and loading different embeddings.
- __custom_env.py__ – defines two types of custom Gym environments:

    - __RecommendationEnv__ – simulates a news recommendation scenario where the user’s state is represented by an averaged embedding of their history. Actions are embedding vectors of recommended articles. The environment computes rewards based on the similarity between the user state and a weighted average of the recommended article embeddings.

    - __RecurrentRecommendationEnv__ – an alternative version where the user state is represented as a sequence of article embeddings (of length MAX_HISTORY). This format supports GRU-based feature extractors to capture temporal patterns in user behavior, enabling richer sequential modeling.

Additionally, the repo provides two demo notebooks for training:

- one using a GRU-based environment with A2C.
- one using the standard environment with DDPG.

## Note

If a Jupyter notebook in the /research folder doesn’t render properly on GitHub, you can view it using [nbviewer](https://nbviewer.org/). Just copy the notebook’s GitHub URL and paste it into the field on the nbviewer page.