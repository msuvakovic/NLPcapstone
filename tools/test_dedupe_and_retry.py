import sys
import os
# ensure package root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join("prompt_opt_harness", "prompt_opt_harness")))
from prompt_opt_harness.llm_backends import MockBackend
from prompt_opt_harness.optimizers.base import evaluate, evaluate_per_example, Candidate

class Ex:
    def __init__(self, text, label):
        self.text = text
        self.label = label

# small lookup mapping for MockBackend
lookup = {
    "Good movie": ("Positive", "movie_reviews"),
    "Bad movie": ("Negative", "movie_reviews"),
}

mock = MockBackend(lookup=lookup)
examples = [Ex("Good movie", "Positive"), Ex("Bad movie", "Negative"), Ex("Good movie", "Positive")]

print("Running evaluate()...")
score = evaluate(mock, "Classify the sentiment of the following text as Positive or Negative.", examples)
print("Score:", score)

print("Running evaluate_per_example()...")
res = evaluate_per_example(mock, "Classify the sentiment of the following text as Positive or Negative.", examples)
for ex, pred, ok in res:
    print(ex.text, pred, ok)

print("Done")
