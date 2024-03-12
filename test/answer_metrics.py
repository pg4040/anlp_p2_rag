from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

# Answer Evaluation Metrics
def evaluate_answer(predicted_answers, true_answers):
    # Ensure both lists contain the same length of elements
    assert len(predicted_answers) == len(true_answers), "Mismatched lengths between predicted and true answers."
    
    f1 = f1_score(true_answers, predicted_answers, average='binary')
    em = np.mean((np.array(predicted_answers) == np.array(true_answers)).astype(int))
    rec = recall_score(true_answers, predicted_answers, average='binary')
    
    return {'f1': f1, 'em': em, 'recall': rec}

# Retrieval Evaluation Metrics
def evaluate_retrieval(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary')
    rec = recall_score(y_true, y_pred, average='binary')
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec}
  
