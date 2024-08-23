# # genaievaluation/genaievaluation/metrics_utils.py
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import BertTokenizer, BertModel
from .embedding_utils import generate_embeddings

# Method to calculate cosine similarity
def calculate_cosine_similarity(embeddings1, embeddings2):
    """
    Calculate cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1 (np.ndarray): Embeddings for ground truth.
        embeddings2 (np.ndarray): Embeddings for generated responses.
    
    Returns:
        np.ndarray: Cosine similarity matrix.
    """
    return cosine_similarity(embeddings1, embeddings2)

# Method to calculate BLEU score
def calculate_bleu_score(reference, hypothesis):
    """
    Calculate BLEU score for a single pair of reference and generated text,
    focusing on bi-grams and applying a smoothing function.
    
    Args:
        reference (str): The ground truth text.
        hypothesis (str): The generated response text.
    
    Returns:
        float: BLEU score.
    """
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    smoothie = SmoothingFunction().method4  # Applying a smoothing method
    score = sentence_bleu(reference_tokens, hypothesis_tokens, weights=(0.5, 0.5), smoothing_function=smoothie)
    return score

# Method to calculate ROUGE score
def calculate_rouge_score(reference, hypothesis):
    """
    Calculate ROUGE score for a single pair of reference and generated text.
    
    Args:
        reference (str): Ground truth text.
        hypothesis (str): Generated text.
        
    Returns:
        dict: ROUGE scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

# Method to calculate BERT similarity
def calculate_bert_similarity(reference, hypothesis):
    """
    Calculate semantic similarity using BERT embeddings.
    
    Args:
        reference (str): The ground truth text.
        hypothesis (str): The generated response text.
    
    Returns:
        float: Cosine similarity score.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    def get_embedding(text):
        try:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()
        except Exception as e:
            print(f"Error in getting BERT embedding: {e}")
            return np.zeros((1, 768))  # Assuming BERT's output dimension is 768
    
    ref_embedding = get_embedding(reference)
    hyp_embedding = get_embedding(hypothesis)
    
    similarity = cosine_similarity(ref_embedding, hyp_embedding)[0][0]
    return similarity

# Method to run evaluation on the entire dataset
def compute_data(df):
    """
    Calculates evaluation metrics and attaches new columns to the pandas dataframe.

    Args:
        df (pd.DataFrame): The DataFrame to be evaluated.

    Returns:
        pd.DataFrame: A DataFrame containing the additional columns apart from 'id', 'prompt', 
                      'ground_truth', and 'generated_response' as shown below:
                      'similarity' : float
                      'bleu_score' : float
                      'bert_similarity' : float
                      'rouge_scores' : dict
    """
    # Check if necessary columns are present
    required_columns = ['id', 'prompt', 'ground_truth', 'generated_response']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Cosine Similarity
    try:
        ground_truth_embeddings = generate_embeddings(df['ground_truth'].tolist())
        generated_response_embeddings = generate_embeddings(df['generated_response'].tolist())
        df['similarity'] = cosine_similarity(ground_truth_embeddings, generated_response_embeddings).diagonal()
    except Exception as e:
        print(f"Error in calculating cosine similarity: {e}")

    # BLEU Score
    try:
        df['bleu_score'] = df.apply(lambda row: calculate_bleu_score(row['ground_truth'], row['generated_response']), axis=1)
    except Exception as e:
        print(f"Error in calculating BLEU score: {e}")

    # BERT Score
    try:
        df['bert_similarity'] = df.apply(lambda row: calculate_bert_similarity(row['ground_truth'], row['generated_response']), axis=1)
    except Exception as e:
        print(f"Error in calculating BERT similarity: {e}")

    # ROUGE Score
    try:
        rouge_scores = [calculate_rouge_score(ref, hyp) for ref, hyp in zip(df['ground_truth'], df['generated_response'])]
        df['rouge_scores'] = rouge_scores
    except Exception as e:
        print(f"Error in calculating ROUGE scores: {e}")

    return df