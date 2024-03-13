def compute_metrics_overall(queries):
    total_accuracy = 0
    total_recall = 0
    total_precision = 0
    total_f1_score = 0
    num_queries = len(queries)

    for query in queries:
        retrieved_documents = query["retrieved_documents"]
        relevant_documents = query["relevant_documents"]

        accuracy = compute_accuracy_per_query(retrieved_documents, relevant_documents)
        recall = compute_recall_per_query([retrieved_documents[0]], relevant_documents)
        precision = compute_precision_per_query(retrieved_documents, relevant_documents)
        f1_score = compute_f1_score_per_query(retrieved_documents, relevant_documents)

        total_accuracy += accuracy
        total_recall += recall
        total_precision += precision
        total_f1_score += f1_score

    average_accuracy = total_accuracy / num_queries
    average_recall = total_recall / num_queries
    average_precision = total_precision / num_queries
    average_f1_score = total_f1_score / num_queries

    return {
        "average_accuracy": average_accuracy,
        "average_recall": average_recall,
        "average_precision_at_one": average_precision,
        "average_f1_score": average_f1_score
    }

def compute_accuracy_per_query(retrieved_documents, relevant_documents):
    total_retrieved = len(retrieved_documents)
    relevant_retrieved = sum(1 for doc in retrieved_documents if doc in relevant_documents)
    if total_retrieved == 0:
        return 0
    return relevant_retrieved / total_retrieved

def compute_recall_per_query(retrieved_documents, relevant_documents):
    total_relevant = len(relevant_documents)
    relevant_retrieved = sum(1 for doc in retrieved_documents if doc in relevant_documents)
    if total_relevant == 0:
        return 0
    return relevant_retrieved / total_relevant

def compute_f1_score_per_query(retrieved_documents, relevant_documents):
    precision = compute_precision_per_query(retrieved_documents, relevant_documents)
    recall = compute_recall_per_query(retrieved_documents, relevant_documents)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def compute_precision_per_query(retrieved_documents, relevant_documents):
    total_retrieved = len(retrieved_documents)
    relevant_retrieved = sum(1 for doc in retrieved_documents if doc in relevant_documents)
    if total_retrieved == 0:
        return 0
    return relevant_retrieved / total_retrieved

