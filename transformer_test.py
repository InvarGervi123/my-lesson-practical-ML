from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("I really enjoy learning machine learning.")

print(result)