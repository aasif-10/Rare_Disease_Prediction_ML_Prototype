import pandas as pd
from sklearn.model_selection import train_test_split

# Load full dataset
df = pd.read_csv("data/rare_diseases_dataset.csv")

# Split into 80% train, 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['disease'])

# Save the train and test sets
train_df.to_csv("data/rare_diseases_dataset_train.csv", index=False)
test_df.to_csv("data/rare_diseases_dataset_test.csv", index=False)

print("✅ Train and Test datasets created!")
