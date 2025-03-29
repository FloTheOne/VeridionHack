import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Încarcă dataset-ul principal
df = pd.read_csv("nouns_only.csv")
df['word'] = df['word'].str.lower()

# Lista cuvintelor principale (main)
main_words = [
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
main_words = [w.strip().lower() for w in main_words]

# Separa cuvintele "main" și celelalte
df_main = df[df['word'].isin(main_words)].copy()
df_rest = df[~df['word'].isin(main_words)].copy()

# Vectorizează definițiile
vectorizer = TfidfVectorizer().fit(df['definition'])
main_vecs = vectorizer.transform(df_main['definition'])
rest_vecs = vectorizer.transform(df_rest['definition'])

# Compară definițiile
matches = []
print("Matching based on definition similarity...")

for i in tqdm(range(len(df_rest))):
    sims = cosine_similarity(rest_vecs[i], main_vecs)[0]
    best_idx = sims.argmax()
    best_score = sims[best_idx]
    matched_main_word = df_main.iloc[best_idx]['word']
    matches.append((df_rest.iloc[i]['word'], matched_main_word, best_score))

# Salvează într-un nou DataFrame
result_df = pd.DataFrame(matches, columns=["word_from_dataset", "matched_main_word", "similarity_score"])

# Exportă rezultatul
result_df.to_csv("matched_words_by_description2.0.csv", index=False, encoding="utf-8")
print(" Result saved to: matched_words_by_description2.0.csv")
