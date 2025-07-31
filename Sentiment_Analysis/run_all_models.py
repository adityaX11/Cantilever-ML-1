#!/usr/bin/env python3
"""
Main script to run all sentiment analysis models and generate comprehensive results.
"""

import os
import sys
import time
from datetime import datetime

def run_svm_model():
    """Run SVM sentiment analysis"""
    print("=" * 50)
    print("RUNNING SVM SENTIMENT ANALYSIS")
    print("=" * 50)
    
    start_time = time.time()
    os.system("python src/train_svm.py")
    end_time = time.time()
    
    print(f"\nSVM training completed in {end_time - start_time:.2f} seconds")
    return end_time - start_time

def run_visualization():
    """Run visualization analysis"""
    print("=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)
    
    start_time = time.time()
    os.system("python src/visualize_results.py")
    end_time = time.time()
    
    print(f"\nVisualization completed in {end_time - start_time:.2f} seconds")
    return end_time - start_time

def run_rnn_model():
    """Run RNN sentiment analysis"""
    print("=" * 50)
    print("RUNNING RNN SENTIMENT ANALYSIS")
    print("=" * 50)
    
    start_time = time.time()
    os.system("python src/train_rnn.py")
    end_time = time.time()
    
    print(f"\nRNN training completed in {end_time - start_time:.2f} seconds")
    return end_time - start_time

def main():
    """Main function to run all models"""
    print("🎬 MOVIE REVIEW SENTIMENT ANALYSIS PROJECT")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if data exists
    if not os.path.exists('data/movie_reviews.csv'):
        print("❌ Error: data/movie_reviews.csv not found!")
        print("Please ensure the dataset exists before running.")
        return
    
    # Check if required directories exist
    for dir_name in ['src', 'data', 'notebooks']:
        if not os.path.exists(dir_name):
            print(f"❌ Error: {dir_name}/ directory not found!")
            return
    
    total_start_time = time.time()
    
    try:
        # Run SVM model
        svm_time = run_svm_model()
        
        # Run visualizations
        viz_time = run_visualization()
        
        # Run RNN model (optional - can be skipped if TensorFlow not available)
        try:
            rnn_time = run_rnn_model()
        except Exception as e:
            print(f"⚠️  RNN model skipped: {e}")
            print("   (TensorFlow might not be installed)")
            rnn_time = 0
        
        total_time = time.time() - total_start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 EXECUTION SUMMARY")
        print("=" * 60)
        print(f"SVM Training:     {svm_time:.2f} seconds")
        print(f"Visualization:    {viz_time:.2f} seconds")
        if rnn_time > 0:
            print(f"RNN Training:     {rnn_time:.2f} seconds")
        print(f"Total Time:       {total_time:.2f} seconds")
        print()
        
        print("📁 Generated Files:")
        print("   - sentiment_distribution.png")
        print("   - review_length_analysis.png")
        print("   - feature_importance.png")
        print("   - confusion_matrix.png")
        if rnn_time > 0:
            print("   - rnn_training_history.png")
            print("   - rnn_sentiment_model.h5")
        
        print("\n🎉 All models completed successfully!")
        print("\nNext steps:")
        print("1. Open notebooks/sentiment_analysis_exploration.ipynb for interactive analysis")
        print("2. Check the generated PNG files for visualizations")
        print("3. Try different preprocessing techniques")
        print("4. Add more data for better results")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main() 