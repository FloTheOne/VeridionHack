import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load GloVe

def load_glove_model(glove_file="glove.6B.100d.txt"):
    model = {}
    with open(glove_file, encoding="utf8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            model[word] = vector
    return model

def load_resources():
    glove = load_glove_model()
    df_excel = pd.read_csv("CuvinteKeycsv.csv", encoding="ISO-8859-1")
    df_excel['Word'] = df_excel['Word'].str.lower()
    df = pd.read_csv("dict.csv")
    df['word'] = df['word'].str.lower()
    return glove, df_excel, df

noun_categories = {
    "object": ["hammer", "rock", "candle", "paper", "gun", "ice", "volcano", "moon", "sword", "shield", "pebble", "robot", "laser", "nuclear"],
    "living_being": ["lion", "bacteria", "virus", "whale", "human", "bird", "tree", "dog", "plant", "insect"],
    "feeling": ["love", "happiness", "sadness", "anger", "peace", "hope", "fear", "excitement", "envy", "gratitude"],
    "event": ["war", "earthquake", "tsunami", "storm", "pandemic", "flood", "explosion", "revolution", "party", "conference"]
}

categories = [
    "feather", "coal", "pebble", "leaf", "paper", "rock", "water", "twig", "sword", "shield", "gun", "flame", "rope", "disease", "cure", "bacteria",
    "shadow", "light", "virus", "sound", "time", "fate", "earthquake", "storm", "vaccine", "logic", "gravity", "robots", "stone", "echo", "thunder",
    "karma", "wind", "ice", "sandstorm", "laser", "magma", "peace", "explosion", "war", "enlightenment", "nuclear bomb", "volcano", "whale", "earth",
    "moon", "star", "tsunami", "supernova", "antimatter", "plague", "rebirth", "tectonic shift", "gamma-ray burst", "human spirit", "apocalyptic meteor",
    "earth’s core", "neutron star", "supermassive black hole", "entropy"
]

def cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def categorize_with_glove(word, glove, threshold=0.5):
    word = word.lower()
    if word not in glove:
        return None, 0.0
    wvec = glove[word]
    best_cat, best_score = None, 0.0
    for cat, sample_words in noun_categories.items():
        scores = [cosine_sim(wvec, glove[ref]) for ref in sample_words if ref in glove]
        if scores:
            avg = np.mean(scores)
            if avg > best_score:
                best_score, best_cat = avg, cat
    return (best_cat, best_score) if best_score >= threshold else (None, best_score)

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
    for cat, words in noun_categories.items():
        if best_label in words:
            return cat, best_score
    return "unknown", best_score

def get_weakness(word, category, df_definitions, df_excel):
    if category in ["object", "living_being", "event"]:
        return {"object": "flame", "living_being": "gun", "event": "time"}[category]
    row = df_definitions[df_definitions['word'] == word]
    if row.empty:
        return "unknown"
    definition = row.iloc[0]['definition']
    excel_words = df_excel['Word'].tolist()
    excel_defs_df = df_definitions[df_definitions['word'].isin(excel_words)]
    excel_defs = excel_defs_df['definition'].tolist()
    vectorizer = TfidfVectorizer().fit([definition] + excel_defs)
    target_vec = vectorizer.transform([definition])
    ref_vecs = vectorizer.transform(excel_defs)
    sims = cosine_similarity(target_vec, ref_vecs)[0]
    best_index = sims.argmax()
    best_word = excel_defs_df.iloc[best_index]['word']
    matched_row = df_excel[df_excel['Word'] == best_word]
    if not matched_row.empty:
        return matched_row.iloc[0]['Weakness']
    return "unknown"

def get_best_word(word, glove, df_excel, df):
    category, score = categorize_with_glove(word, glove)
    if not category:
        category, score = categorize_with_tfidf(word, df)
    weakness = get_weakness(word, category, df, df_excel)
    return weakness

# Example usage:
# glove, df_excel, df = load_resources()
# print(get_best_word("tornado", glove, df_excel, df))
