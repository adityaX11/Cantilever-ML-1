# Movie Review Sentiment Analysis

A comprehensive sentiment analysis project that classifies movie reviews as positive or negative using a traditional machine learning (SVM) approach.

## 🎯 Project Overview

This project demonstrates:

- **Text preprocessing** using NLTK (tokenization, lemmatization, stopword removal)
- **Feature extraction** using TF-IDF vectorization
- **Machine Learning** using Support Vector Machine (SVM)
- **Data visualization** and analysis
- **Interactive exploration** via Jupyter notebooks

## 🛠️ Technologies Used

- **Python** - Core programming language
- **NLTK** - Natural language processing and text preprocessing
- **scikit-learn** - Machine learning (SVM, TF-IDF)
- **pandas** - Data manipulation
- **matplotlib & seaborn** - Data visualization
- **Jupyter** - Interactive notebooks

## 📁 Project Structure

```Structure of Prject Application
machinlearning-1/
├── data/
│   └── movie_reviews.csv          # Dataset with movie reviews
├── src/
│   ├── preprocess.py              # Text preprocessing functions
│   ├── features.py                # Feature extraction utilities
│   ├── train_svm.py               # SVM model training
│   ├── train_rnn.py               # RNN model training
│   ├── visualize_results.py       # Data visualization
│   └── test_preprocess.py         # Preprocessing test script
├── notebooks/
│   └── sentiment_analysis_exploration.ipynb  # Interactive analysis
├── requirements.txt               # Python dependencies
├── run_all_models.py              # Main execution script
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run All Models

```bash
python run_all_models.py
```

This will:

- Train the SVM model
- Generate visualizations
- Save all results and plots

### 3. Individual Model Execution

**SVM Model:**

```bash
python src/train_svm.py
```

**Visualizations:**

```bash
python src/visualize_results.py
```

**Test Preprocessing:**

```bash
python src/test_preprocess.py
```

### Console Output

- Classification reports (precision, recall, F1-score)
- Model training progress
- Sample predictions

### Generated Files

- `sentiment_distribution.png` - Sentiment class distribution
- `review_length_analysis.png` - Review length analysis
- `feature_importance.png` - Most important words
- `confusion_matrix.png` - Model performance visualization

## 🔍 Interactive Analysis

Open the Jupyter notebook for interactive exploration:

```bash
jupyter notebook notebooks/sentiment_analysis_exploration.ipynb
```

The notebook includes:

- Data exploration and visualization
- Preprocessing analysis
- Model comparison
- Interactive prediction testing

## 📈 Model Performance

### SVM Model

- Uses TF-IDF features
- Fast training and prediction
- Good baseline performance
- Interpretable feature importance

### RNN Model (currently this is not work due to dependencies)

- Uses word embeddings and LSTM layers
- Captures sequential information
- Better for complex language patterns
- Requires more training time

## 🎯 Key Features

### Text Preprocessing

- **Lowercase conversion**
- **Punctuation removal**
- **Tokenization** using NLTK
- **Lemmatization** for word normalization
- **Stopword removal**

### Feature Engineering

- **TF-IDF vectorization** for SVM
- **Word embeddings** for RNN
- **Sequence padding** for neural networks

### Model Evaluation

- **Classification reports** (precision, recall, F1-score)
- **Confusion matrices**
- **Training history plots** (for RNN)
- **Feature importance analysis**

## 🔧 Customization

### Adding More Data

Edit `data/movie_reviews.csv` to add more reviews:

```csv
review,sentiment
"Your review text here",positive
"Another review text",negative
```

### Modifying Preprocessing

Edit `src/preprocess.py` to:

- Add stemming
- Include n-gram features
- Customize stopwords

### Experimenting with Models

- Try different SVM kernels
- Adjust RNN architecture
- Experiment with different embedding dimensions

## 🐛 Troubleshooting

### NLTK Resource Errors

If you encounter NLTK resource errors:

- The script automatically downloads required resources
- Check internet connection
- Manual download: `python -c "import nltk; nltk.download('all')"`

### TensorFlow Issues

If RNN model fails:

- Ensure TensorFlow is installed: `pip install tensorflow`
- Check GPU compatibility
- The SVM model will still work without TensorFlow

### Memory Issues

For large datasets:

- Reduce `max_features` in TF-IDF
- Decrease `max_words` in RNN
- Use smaller batch sizes

## 📚 Learning Resources

- [NLTK Documentation](https://www.nltk.org/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Sentiment Analysis Best Practices](https://towardsdatascience.com/sentiment-analysis-best-practices-8e8c3c0b3b4c)

## 🤝 Contributing

Feel free to:

- Add more preprocessing techniques
- Implement additional models
- Improve visualizations
- Add more datasets

## 📄 License

This project is for educational purposes. Feel free to use and modify as needed.

---

## 🌐 Web-based GUI

A modern web app for sentiment prediction and visualization using Streamlit.
In this there are two features

->1.Text formate csv file use.
->2.Can upload csv file from Browser.

### Run the Web App

```bash
streamlit run src/web_app.py
```

- Enter a movie review and get instant prediction (SVM)
- View all generated visualizations in your browser.
