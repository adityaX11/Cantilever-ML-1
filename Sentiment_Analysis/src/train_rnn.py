import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from preprocess import preprocess
import joblib

def load_and_prepare_data():
    """Load and prepare data for RNN"""
    df = pd.read_csv('data/movie_reviews.csv')
    df['clean_review'] = df['review'].apply(preprocess)
    
    # Convert sentiment to numeric
    df['sentiment_numeric'] = df['sentiment'].map({'negative': 0, 'positive': 1})
    
    return df

def prepare_sequences(texts, max_words=1000, max_len=100):
    """Convert text to sequences for RNN"""
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return padded_sequences, tokenizer

def build_rnn_model(max_words, max_len, embedding_dim=16):
    """Build RNN model architecture"""
    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('rnn_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to train RNN model"""
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    # Prepare sequences
    print("Converting text to sequences...")
    X, tokenizer = prepare_sequences(df['clean_review'])
    y = df['sentiment_numeric'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build model
    print("Building RNN model...")
    max_words = 1000
    max_len = 100
    model = build_rnn_model(max_words, max_len)
    
    print("Model Summary:")
    model.summary()
    
    # Train model
    print("Training RNN model...")
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    # Save model
    model.save('rnn_sentiment_model.h5')
    joblib.dump(tokenizer, 'rnn_tokenizer.joblib')
    print("Model saved as 'rnn_sentiment_model.h5' and tokenizer as 'rnn_tokenizer.joblib'")
    
    # Test with sample predictions
    print("\nSample Predictions:")
    sample_texts = [
        "I loved this movie, it was fantastic!",
        "Terrible film. Waste of time.",
        "The acting was outstanding and the plot was engaging."
    ]
    
    for text in sample_texts:
        cleaned = preprocess(text)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        print(f"Text: '{text}'")
        print(f"Prediction: {sentiment} (confidence: {confidence:.2f})")
        print()

if __name__ == "__main__":
    main() 