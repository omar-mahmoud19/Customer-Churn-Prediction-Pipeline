import pandas as pd

df = pd.read_csv('data/processed/clean_data.csv')
sample = df.sample(n=5000, random_state=42)
sample.to_csv('data/processed/clean_data_sample.csv', index=False)
print('Done:', sample.shape)