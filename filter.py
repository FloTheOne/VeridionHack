import pandas as pd

# Load your dataset
df = pd.read_csv('./dict.csv')

# Filter rows explicitly marked as nouns
df_nouns = df[df['definition'].str.contains(r'Noun', case=False, na=False)]

# Save the filtered nouns dataset
df_nouns.to_csv('./nouns_only.csv', index=False)
