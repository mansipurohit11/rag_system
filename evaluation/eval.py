from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# Build a test dataset (question, ground truth, answer, contexts)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
print(result)  # scores 0-1 for each metric