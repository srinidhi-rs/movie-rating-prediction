# Imports
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Jupyter inline plotting magic - remove this line for script-based use
# %matplotlib inline

# Load datasets
users_df = pd.read_csv('users.dat', sep='::', names=['UserID','Gender','Age','Occupation','zip-code'], engine='python')
ratings_df = pd.read_csv('ratings.dat', sep='::', names=['UserID','MovieID','Rating','Timestamp'], parse_dates=['Timestamp'], engine='python')
movies_df = pd.read_csv('movies.dat', sep='::', names=['MovieID','Title','Genres'], engine='python', encoding='ISO-8859-1')

# Merge datasets
movie_ratings_df = pd.merge(movies_df, ratings_df, on='MovieID')
movie_ratings_users_df = pd.merge(movie_ratings_df, users_df, on='UserID')

# Create Master_Data
Master_Data = movie_ratings_users_df.drop(['zip-code', 'Timestamp'], axis=1)

# Visualization 1: User Age Distribution
plt.figure(figsize=(8,6))
Master_Data['Age'].hist()
plt.title('User Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Users')
plt.show()

# Visualization 2: User rating of the movie “Toy Story”
plt.figure(figsize=(8,6))
user_rating = Master_Data[Master_Data.Title == "Toy Story (1995)"]
user_rating['Rating'].hist()
plt.title('User rating of the movie “Toy Story”')
plt.xlabel('Rating')
plt.ylabel('Number of Users')
plt.show()

# Visualization 3: Top 25 movies by viewership rating
dfTop25 = Master_Data.groupby('Title').size().sort_values(ascending=False)[:25]
dfTop25.plot(kind='barh', alpha=0.6, figsize=(7,7))
plt.xlabel("Viewership Ratings Count")
plt.ylabel("Movies (Top 25)")
plt.title("Top 25 movies by viewership rating")
plt.show()

# Visualization 4: Ratings by user id = 2696
user_2696 = Master_Data[Master_Data['UserID'] == 2696]
print(user_2696.head(2))

# Feature Engineering - Genres
dfGenres = Master_Data['Genres'].str.split("|")
listGenres = set()
for genre in dfGenres:
    listGenres = listGenres.union(set(genre))

# One-hot encoding of genres
ratingsOneHot = Master_Data['Genres'].str.get_dummies("|")
Master_Data = pd.concat([Master_Data, ratingsOneHot], axis=1)

# Extract year from title
Master_Data[["Title","Year"]] = Master_Data.Title.str.extract("(.*)\s\((\d{4})\)", expand=True)
Master_Data['Year'] = Master_Data['Year'].astype(int)

# Encode Gender
Master_Data['Gender'] = Master_Data['Gender'].replace({'F': 1, 'M': 0}).astype(int)

# Gender, Age, Occupation vs Rating
Master_Data.groupby(["Gender","Rating"]).size().unstack().plot(kind='bar', stacked=False)
plt.title("Rating distribution by Gender")
plt.show()

Master_Data.groupby(["Age","Rating"]).size().unstack().plot(kind='bar', stacked=False)
plt.title("Rating distribution by Age")
plt.show()

Master_Data.groupby(["Occupation","Rating"]).size().unstack().plot(kind='bar', stacked=False)
plt.title("Rating distribution by Occupation")
plt.show()

Master_Data.groupby(["Year","Rating"]).size().unstack().plot(kind='bar', stacked=False, figsize=(12,6))
plt.title("Rating distribution by Year")
plt.show()

# Feature Selection using RFE
X = Master_Data[Master_Data.columns.difference(['Rating', 'Title', 'Genres'])]
y = Master_Data['Rating']
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
rfe.fit(X, y)

# Print selected features
for i in range(X.shape[1]):
    print('Column: %d %s, Selected=%s, Rank: %d' % (i, X.columns[i], rfe.support_[i], rfe.ranking_[i]))

# Linear Regression Model
Master_Data_sample = Master_Data.sample(n=50000, random_state=0)
X = Master_Data_sample[['Gender','Age','Occupation']]
y = Master_Data_sample['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)

# Evaluation
print('y-intercept: ', linear_reg.intercept_)
print('Beta coefficients: ', linear_reg.coef_)
print('Mean Abs Error  MAE: ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Sq Error  MSE: ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Sq Error RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r2 value: ', metrics.r2_score(y_test, y_pred))

# Show prediction vs test
prediction_df = pd.DataFrame({'Test': y_test, 'Prediction': y_pred})
print(prediction_df.head())
