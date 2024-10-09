# Fake_News_Prediction
Machine Learning project for fake news prediction
# 2 Dataset are uploaded "Fake" and "True".
# I have combined both to make a new dataset named "News". i was unable to upload that since the file was large even after it is compressed.
# Codes used to combine the datasets is given below
import pandas as pd

# Load the True and Fake news datasets
true_news_df = pd.read_csv(r"C:\Users\akash\Downloads\Untitled (1)\True.csv")
fake_news_df = pd.read_csv(r"C:\Users\akash\Downloads\Untitled (1)\Fake.csv")

# Add a 'label' column to differentiate between True (1) and Fake (0) news
true_news_df['label'] = 1
fake_news_df['label'] = 0

# Combine the two datasets
news_df = pd.concat([true_news_df, fake_news_df], axis=0).reset_index(drop=True)

# Check the combined dataset
news_df.head()

# Shuffled the new dataset entries for model accuracy
