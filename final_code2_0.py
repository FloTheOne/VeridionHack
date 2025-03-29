import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api

# Step 1: Load GloVe from gensim API
print("Loading GloVe from gensim...")
glove = api.load("glove-wiki-gigaword-100")  # returns KeyedVectors
print("GloVe loaded!")

# Step 2: Manual categories
noun_categories = {
    "object": [
        "hammer", "rock", "candle", "paper", "gun", "ice", "volcano", "moon",
        "sword", "shield", "pebble", "robot", "laser", "nuclear"
    ],
    "living_being": [
        "lion", "bacteria", "virus", "whale", "human", "bird", "tree", "dog", "plant", "insect"
    ],
    "feeling": [
        "love", "happiness", "sadness", "anger", "peace", "hope", "fear", "excitement", "envy", "gratitude"
    ],
    "event": [
        "war", "earthquake", "tsunami", "storm", "pandemic", "flood", "explosion", "revolution", "party", "conference"
    ]
}

# 🔹 Main word list for fallback TF-IDF matching
categories = [
    "feather", "coal", "pebble", "leaf", "paper", "rock", "water", "twig",
    "sword", "shield", "gun", "flame", "rope", "disease", "cure", "bacteria",
    "shadow", "light", "virus", "sound", "time", "fate", "earthquake", "storm",
    "vaccine", "logic", "gravity", "robots", "stone", "echo", "thunder", "karma",
    "wind", "ice", "sandstorm", "laser", "magma", "peace", "explosion", "war",
    "enlightenment", "nuclear bomb", "volcano", "whale", "earth", "moon", "star",
    "tsunami", "supernova", "antimatter", "plague", "rebirth", "tectonic shift",
    "gamma-ray burst", "human spirit", "apocalyptic meteor", "earth’s core",
    "neutron star", "supermassive black hole", "entropy"
]

# Step 3: Categorize using GloVe vectors from gensim
def categorize_with_glove(word, threshold=0.5):
    word = word.lower()
    if word not in glove:
        return None, 0.0

    best_cat = None
    best_score = 0.0

    for cat, sample_words in noun_categories.items():
        scores = []
        for ref in sample_words:
            if ref in glove:
                score = glove.similarity(word, ref)
                scores.append(score)
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_cat = cat

    if best_score >= threshold:
        return best_cat, best_score
    return None, best_score

# Step 4: Fallback with TF-IDF and definitions
def categorize_with_tfidf(word, df):
    row = df[df['word'] == word]
    if row.empty:
        return "unknown", 0.0

    definition = row.iloc[0]['definition']
    df_main = df[df['word'].isin(categories)]

    main_definitions = df_main['definition'].tolist()
    main_labels = df_main['word'].tolist()

    vectorizer = TfidfVectorizer().fit([definition] + main_definitions)
    def_vec = vectorizer.transform([definition])
    main_vecs = vectorizer.transform(main_definitions)

    sims = cosine_similarity(def_vec, main_vecs)[0]
    best_index = sims.argmax()
    best_label = main_labels[best_index]
    best_score = sims[best_index]

    # Map best match to its category if it exists
    for cat, words in noun_categories.items():
        if best_label in words:
            return cat, best_score
    return "unknown", best_score

# Step 5: Combined categorization
def categorize(word, df, glove_threshold=0.5):
    category, score = categorize_with_glove(word, threshold=glove_threshold)
    if category:
        return category, score
    return categorize_with_tfidf(word, df)

# 📘 Load definitions dataset
df = pd.read_csv("dict.csv")
df['word'] = df['word'].str.lower()

# 🧪 Example test
test_words = ["clergy", "eruption", "happiness", "gorilla", "tornado", "rocket", "sympathy", "hero"]

for w in test_words:
    cat, conf = categorize(w, df)
    print(f"{w:10} → {cat:15} (confidence: {conf:.2f})")
