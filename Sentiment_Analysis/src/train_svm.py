import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from preprocess import preprocess
from features import extract_tfidf_features, save_svm_model_and_vectorizer

# Check current working directory
print("Current working directory:", os.getcwd())

#Correct relative path from script location
csv_path = '../data/movie_reviews.csv'

# Check file existence before loading
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File not found: {csv_path}")
else:
    print(f"Found CSV: {csv_path}")

#Load dataset
df = pd.read_csv(csv_path)

#Preprocess
print('Preprocessing...')
df['clean_review'] = df['review'].apply(preprocess)

#Feature Extraction
print('Extracting features...')
X, vectorizer = extract_tfidf_features(df['clean_review'])
y = df['sentiment']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
print('Training SVM...')
clf = LinearSVC()
clf.fit(X_train, y_train)

#  Save model and vectorizer
save_svm_model_and_vectorizer(clf, vectorizer)

#  Evaluate
print('Evaluating...')
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
