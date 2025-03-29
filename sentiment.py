import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from tqdm import tqdm

# Load GloVe model
print("Loading GloVe model...")
glove_model = api.load("glove-wiki-gigaword-100")

# Load your CSV
df = pd.read_csv("nouns_only.csv")
df['word'] = df['word'].str.lower()

# Define emotional anchor words
sentiment_anchors = [
    "happy", "joy", "love", "fear", "sad", "anger", "hope", "grief", "hate",
    "calm", "depression", "rage", "delight", "panic", "disgust", "lonely"
]

# Filter out anchors not in GloVe
valid_anchors = [w for w in sentiment_anchors if w in glove_model]
anchor_vecs = np.array([glove_model[w] for w in valid_anchors])

# Function to check if a word is sentiment-related
def is_sentiment_like(word, threshold=0.4):
    if word not in glove_model:
        return False
    vec = glove_model[word].reshape(1, -1)
    sims = cosine_similarity(vec, anchor_vecs)[0]
    return max(sims) > threshold

# Apply to your dataset
print("Labeling sentiment-related words...")
tqdm.pandas()
df["is_sentiment"] = df["word"].progress_apply(is_sentiment_like)

# Save results
df.to_csv("labeled_sentiment_words.csv", index=False)
print("Saved labeled file as labeled_sentiment_words.csv ✅")
