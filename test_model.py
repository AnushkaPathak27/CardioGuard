import pickle

with open("classifier.pkl", "rb") as file:  # Replace with your actual model filename
    model = pickle.load(file)

print("Model loaded successfully!")
