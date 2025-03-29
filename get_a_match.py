import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load dataset
df = pd.read_csv("dict.csv")
df['word'] = df['word'].str.lower()

# 2. Define your main words
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
main_words = [w.lower().strip() for w in main_words]

# 3. Choose the word you want to classify
target_word = "clergy"

# 4. Extract its definition
target_row = df[df['word'] == target_word]
if target_row.empty:
    print(f"Word '{target_word}' not found in dataset.")
else:
    target_definition = target_row.iloc[0]['definition']

    # 5. Extract definitions for all main words
    df_main = df[df['word'].isin(main_words)]
    main_definitions = df_main['definition'].tolist()
    main_labels = df_main['word'].tolist()

    # 6. Vectorize the definitions
    vectorizer = TfidfVectorizer().fit([target_definition] + main_definitions)
    target_vec = vectorizer.transform([target_definition])
    main_vecs = vectorizer.transform(main_definitions)

    # 7. Calculate similarity
    similarities = cosine_similarity(target_vec, main_vecs)[0]
    best_index = similarities.argmax()
    best_label = main_labels[best_index]
    best_score = similarities[best_index]

    # 8. Output result
    print(f"\n Word: '{target_word}'")
    print(f"Best match: '{best_label}'")
    print(f"Similarity score: {best_score:.4f}")
