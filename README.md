🎬 MovieLens Data Analysis & Rating Prediction

This project explores the MovieLens dataset to analyze movie preferences, uncover viewing trends, and predict user ratings based on demographic and movie-related features.

📦 Dataset
Data used:

users.dat – Demographic info (age, gender, occupation, zip)

movies.dat – Movie titles and genres

ratings.dat – User ratings for movies

These files were merged into a unified dataset called Master_Data.

🔍 Project Objectives

Merge and clean datasets for analysis

Perform exploratory data analysis (EDA) on user behavior and movie trends

Engineer features from raw data (e.g. extracting genres, years)

Build a predictive model to estimate movie ratings

Evaluate model performance

📊 Exploratory Data Analysis

Visualized:

Age distribution of users

Movie rating trends (e.g. Toy Story (1995) analysis)

Most-watched movies

Top-rated movies with sufficient views

Demographic-based rating tendencies (age, gender, occupation)

🛠️ Feature Engineering

One-hot encoded genres for multi-label classification

Extracted release year from movie titles

Encoded gender (M/F → 1/0)

Filtered movies with less than 100 views for reliability

🤖 Machine Learning

Feature Selection

Used Recursive Feature Elimination (RFE) with a Decision Tree Classifier

Model

Built a Linear Regression model using:

Age

Gender

Occupation

Target: Rating

Evaluation Metrics:

📉 MAE: ~0.89

📉 MSE: ~1.24

📉 RMSE: ~1.11

📈 R² Score: ~0.0036

Note: Model performance suggests potential for deeper modeling using content-based or collaborative filtering techniques.

🧰 Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn


🚀 Future Improvements

Use collaborative filtering or matrix factorization (e.g., SVD)

Build a recommendation system

Apply advanced regression models (Random Forest, XGBoost)

Include timestamp-based trend analysis

📚 References

GroupLens MovieLens Dataset

Feel free to fork or star the repo if you found this helpful! ✨
