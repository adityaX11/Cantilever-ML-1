import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from preprocess import preprocess
from features import extract_tfidf_features
from features import save_svm_model_and_vectorizer

# Load your data (expects columns: 'review', 'sentiment')
# df = pd.read_csv('../data/movie_reviews.csv')
df = pd.read_csv('data/movie_reviews.csv')  # ✅ REMOVE ../



# Preprocess
print('Preprocessing...')
df['clean_review'] = df['review'].apply(preprocess)

# Feature extraction
print('Extracting features...')
X, vectorizer = extract_tfidf_features(df['clean_review'])
y = df['sentiment']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
print('Training SVM...')
clf = LinearSVC()
clf.fit(X_train, y_train)

# Save model and vectorizer
save_svm_model_and_vectorizer(clf, vectorizer)

# Evaluate
print('Evaluating...')
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred)) 