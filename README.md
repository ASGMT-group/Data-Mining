# Collaborative Filtering Recommender System

This project implements a collaborative filtering recommender system using the Singular Value Decomposition (SVD) algorithm from the Surprise library. The system predicts what products a user might be interested in based on their browsing history.

## Table of Contents

- [Introduction](#introduction)
- [How it Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This recommender system is designed to provide personalized recommendations for users based on their browsing history. It uses collaborative filtering, leveraging the SVD algorithm, to uncover patterns in user behavior and make predictions about their preferences.

## How it Works

1. **Data Loading and Merging:**
   - Three datasets are loaded: browsing history, item data, and member data.
   - The datasets are merged based on common columns to create a unified dataset for training the recommender system.

2. **Data Preprocessing:**
   - The merged dataset is prepared for the recommender system.
   - The 'Reader' class from Surprise is used to define the rating scale (binary interactions in this case).

3. **Building the Recommender Model:**
   - The merged dataset is loaded into Surprise's `Dataset` class.
   - The data is split into training and testing sets using the `train_test_split` function.
   - The SVD algorithm is chosen as the collaborative filtering model for training.

4. **Training the Model:**
   - The SVD model is trained on the training set using the `fit` method.

5. **Making Predictions:**
   - For a specific member (user), the system generates recommendations for items (products) based on their browsing history.
   - The recommender predicts the likelihood of a user interacting with an item by estimating the rating.

6. **Getting Top Recommendations:**
   - The predicted ratings for items are sorted in descending order to get the top recommendations for a user.

7. **Displaying Recommendations:**
   - The code prints the top N recommendations along with their predicted ratings for a specific member.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/recommender-system.git](https://github.com/ASGMT-group/Data-Mining.git)https://github.com/ASGMT-group/Data-Mining.git
   cd recommender-system
2. Install dependencies:
  ```bash
  pip install -r requirements1.txt
  #run scripts
  #example
  python recommende.py

