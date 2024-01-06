import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the dataset
browsing_history_path = 'bigBasket.csv'
item_data_path = 'item.csv'
member_data_path = 'Memberdata.csv'

browsing_df = pd.read_csv(browsing_history_path)
item_data_df = pd.read_csv(item_data_path)
member_data_df = pd.read_csv(member_data_path)

# Display the first few rows of each dataset (optional)
print("Browsing History Data:")
print(browsing_df.head())

print("\nItem Data:")
print(item_data_df.head())

print("\nMember Data:")
print(member_data_df.head())

# Merge datasets if needed
merged_df = pd.merge(browsing_df, item_data_df, on=['Member', 'Description'], how='left')
merged_df = pd.merge(merged_df, member_data_df, on=['Member', 'Order', 'SKU', 'Created On', 'Description'], how='left')

# Display the merged dataset (optional)
print("\nMerged Data:")
print(merged_df.head())

# Define the rating scale for Surprise
reader = Reader(rating_scale=(0, 1))  # Assuming binary interactions (0 or 1)

# Load the dataset into Surprise format
data = Dataset.load_from_df(merged_df[['Member', 'Description', 'Order']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Use Singular Value Decomposition (SVD) algorithm
model = SVD()

# Train the model on the training set
model.fit(trainset)

# Make recommendations for a specific user (replace member_id with an actual member ID)
member_id = merged_df['Member'].iloc[0]
user_recommendations = [(item, model.predict(member_id, item).est) for item in merged_df['Description'].unique()]

# Get top N recommendations
top_n = 5
user_recommendations.sort(key=lambda x: x[1], reverse=True)
top_recommendations = user_recommendations[:top_n]

print(f"\nTop {top_n} recommendations for member {member_id} based on browsing history:")
for item, score in top_recommendations:
    print(f"Item: {item}, Predicted Rating = {score}")
    