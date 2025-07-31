from preprocess import preprocess

# Test the preprocessing function
sample = "I loved this movie, it was fantastic! The acting was excellent."
print("Original text:", sample)
cleaned = preprocess(sample)
print("Cleaned text:", cleaned)

# Test with another sample
sample2 = "Terrible film. Waste of time. Boring plot."
print("\nOriginal text:", sample2)
cleaned2 = preprocess(sample2)
print("Cleaned text:", cleaned2)
