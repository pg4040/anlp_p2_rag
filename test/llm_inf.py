import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def prepare_data():
    file1 = '../Faculty_data/qa_pairs/simple-metadata-qa-pairs.csv' #Question,Answer
    file2 = '../Courses_data/qa_pairs/QnA_data_courses.csv' #Question,Ref_Answer
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df1.rename(columns={'Answer':'Ref_Answer'}, inplace=True)
    N = 100
    random_df1 = df1.sample(N)
    random_df2 = df2.sample(N)
    combined_df = pd.concat([random_df1, random_df2[['Question', 'Ref_Answer']]]) 
    return combined_df

def calculate_cosine_similarity_TFIDF(text1, text2):
    """Calculate cosine similarity between two clean texts."""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def distil_BERT_QA(): #TODO - Provide a context - by combining all our documents.  
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    combined_df = prepare_data()
    print(combined_df.head())
    for index, row in combined_df.iterrows():
        question = row['Question']
        true_answer = row['Ref_Answer']
        
        # Use the QA model to generate an answer
        generated_answer = qa_pipeline({
            'question': question,
            'context': ' '  # Provide a relevant context or modify the script to include context if available
        })['answer']
        
        # Compare the generated answer with the true answer
        similarity_score = calculate_cosine_similarity_TFIDF(generated_answer, true_answer)
        print(f"Question: {question}\nGenerated Answer: {generated_answer}\nTrue Answer: {true_answer}\nSimilarity: {similarity_score}\n")

def gpt2_QA():
    generator = pipeline('text-generation', model='gpt2-medium')
    combined_df = prepare_data()
    print(combined_df.head())
    for index, row in combined_df.iterrows():
        question = row['Question']
        true_answer = row['Ref_Answer'].strip()
        answer = generator(question, max_length=50, num_return_sequences=1)[0]
        generated_answer = answer['generated_text'][len(question):].strip()
        similarity_score = "TBD"
        print(f"Question: {question}\nGenerated Answer: {generated_answer}\nTrue Answer: {true_answer}\nSimilarity: {similarity_score}\n")

     
if __name__ == '__main__':
    gpt2_QA()
