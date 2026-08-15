import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# === Load and Prepare Data (MovieLens 1M format with "::" delimiter) ===
ratings = pd.read_csv("ratings.dat", sep="::", engine='python',
                      names=["userId", "movieId", "rating", "timestamp"])

users = pd.read_csv("users.dat", sep="::", engine='python',
                    names=["userId", "gender", "age", "occupation", "zip_code"])

movies = pd.read_csv("movies.dat", sep="::", engine='python', encoding='latin-1',
                     names=["movieId", "title", "genres"])

# Convert ratings to binary "liked" target
ratings['liked'] = (ratings['rating'] >= 4).astype(int)

# Merge datasets
data = ratings.merge(users, on="userId").merge(movies, on="movieId")
data['gender'] = data['gender'].map({'M': 1, 'F': 0})
data = pd.get_dummies(data, columns=['occupation'])

# === Process genres into binary columns ===
genre_cols = data['genres'].str.get_dummies(sep='|')
data = pd.concat([data, genre_cols], axis=1)

# === Feature selection ===
features = ['age', 'gender'] + list(genre_cols.columns) + \
           [col for col in data.columns if col.startswith('occupation_')]

X = data[features]
y = data['liked']

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = LogisticRegression(max_iter=1500, solver='saga', n_jobs=-1)
model.fit(X_train, y_train)

# === Counterfactual Analysis ===
test_indices = X_test.index[:10000]
ground_truth = data.loc[test_indices, 'liked'].reset_index(drop=True)

results = []
for i in range(10000):
    original = X_test.iloc[i].copy()
    counterfactual = original.copy()
    counterfactual['gender'] = 1 - original['gender']  # Flip gender

    # Predict
    original_pred = model.predict([original])[0]
    counterfactual_pred = model.predict([counterfactual])[0]

    # Detect original occupation
    occ_cols = [col for col in X_test.columns if 'occupation_' in col and original[col] == 1]
    occupation = occ_cols[0].replace('occupation_', '') if occ_cols else 'unknown'

    results.append({
        'index': i,
        'age': original['age'],
        'gender': 'Male' if original['gender'] == 1 else 'Female',
        'occupation': occupation,
        'original_pred': original_pred,
        'counterfactual_pred': counterfactual_pred,
        'ground_truth': ground_truth[i],
        'changed': original_pred != counterfactual_pred
    })

cf_df = pd.DataFrame(results)
cf_df['fair_pred'] = np.where(cf_df['changed'], cf_df['counterfactual_pred'], cf_df['original_pred'])
cf_df.to_csv("counterfactual_analysis.csv", index=False)

# === Summary ===
changed_count = cf_df['changed'].sum()
total = len(cf_df)
percent_changed = round((changed_count / total) * 100, 2)

print(f"Generated counterfactual_analysis.csv with {total} entries.")
print(f"Total Predictions Changed: {changed_count}")
print(f"Percentage Changed: {percent_changed}%")
