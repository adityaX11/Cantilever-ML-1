import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from preprocess import preprocess

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare the data for analysis"""
    df = pd.read_csv('data/movie_reviews.csv')
    df['clean_review'] = df['review'].apply(preprocess)
    return df

def plot_sentiment_distribution(df):
    """Plot the distribution of sentiments"""
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment'].value_counts()
    
    plt.subplot(1, 2, 1)
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Sentiment Distribution')
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='sentiment')
    plt.title('Sentiment Count')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_review_length_analysis(df):
    """Analyze review length by sentiment"""
    df['review_length'] = df['review'].str.len()
    df['word_count'] = df['review'].str.split().str.len()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='sentiment', y='review_length')
    plt.title('Review Length by Sentiment')
    plt.ylabel('Character Count')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='sentiment', y='word_count')
    plt.title('Word Count by Sentiment')
    plt.ylabel('Word Count')
    
    plt.tight_layout()
    plt.savefig('review_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(df):
    """Show most important words for each sentiment"""
    # Prepare data
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(df['clean_review'])
    y = df['sentiment']
    
    # Train a simple model to get feature importance
    clf = LinearSVC()
    clf.fit(X, y)
    
    # Get feature names and importance
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = np.abs(clf.coef_[0])
    
    # Get top words for each sentiment
    positive_indices = np.argsort(feature_importance)[-15:]  # Top 15
    negative_indices = np.argsort(feature_importance)[:15]   # Bottom 15
    
    plt.figure(figsize=(15, 6))
    
    # Positive words
    plt.subplot(1, 2, 1)
    positive_words = [feature_names[i] for i in positive_indices]
    positive_scores = [feature_importance[i] for i in positive_indices]
    plt.barh(range(len(positive_words)), positive_scores)
    plt.yticks(range(len(positive_words)), positive_words)
    plt.title('Most Important Words (Positive)')
    plt.xlabel('Feature Importance')
    
    # Negative words
    plt.subplot(1, 2, 2)
    negative_words = [feature_names[i] for i in negative_indices]
    negative_scores = [feature_importance[i] for i in negative_indices]
    plt.barh(range(len(negative_words)), negative_scores)
    plt.yticks(range(len(negative_words)), negative_words)
    plt.title('Most Important Words (Negative)')
    plt.xlabel('Feature Importance')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(df):
    """Train model and plot confusion matrix"""
    # Prepare data
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['clean_review'])
    y = df['sentiment']
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    
    # Predict and create confusion matrix
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def main():
    """Run all visualizations"""
    print("Loading data...")
    df = load_and_prepare_data()
    
    print("Creating visualizations...")
    plot_sentiment_distribution(df)
    plot_review_length_analysis(df)
    plot_feature_importance(df)
    plot_confusion_matrix(df)
    
    print("All visualizations saved as PNG files!")

if __name__ == "__main__":
    main() 