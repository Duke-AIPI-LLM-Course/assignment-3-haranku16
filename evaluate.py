import pandas as pd
import requests
import os
import matplotlib.pyplot as plt
from openai import OpenAI
from scripts.vectordb import VectorDatabase
from scripts.rag import RetrievalAugmentedGenerator

def download_novel():
    """Download The Time Machine if not already present"""
    url = "https://www.gutenberg.org/cache/epub/35/pg35.txt"
    filename = "time_machine.txt"
    
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
    return filename

def evaluate_response(client, generated_response, ground_truth, context):
    """Use GPT-4 to evaluate the response and context"""
    prompt = f"""You are an objective evaluator. Score the following aspects of a RAG system's response:

Ground Truth Question & Answer:
{ground_truth}

Generated Response:
{generated_response}

Retrieved Context:
{context}

Please provide two scores (0-10, decimals allowed):
1. Context Relevance Score: How relevant is the retrieved context to answering the question?
2. Answer Accuracy Score: How accurate is the generated answer compared to the ground truth?

Format your response exactly as follows:
Context Score: [score]
Answer Score: [score]"""

    evaluation = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    response = evaluation.choices[0].message.content
    
    # Parse scores from response
    context_score = float(response.split("Context Score:")[1].split("\n")[0].strip())
    answer_score = float(response.split("Answer Score:")[1].strip())
    
    return context_score, answer_score

def plot_score_distributions(context_scores, answer_scores):
    """Plot distributions of context and answer scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(context_scores, bins=20, alpha=0.7)
    ax1.set_title('Distribution of Context Relevance Scores')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Frequency')
    
    ax2.hist(answer_scores, bins=20, alpha=0.7)
    ax2.set_title('Distribution of Answer Accuracy Scores')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close()

def main():
    # Load ground truth data
    ground_truth = pd.read_csv('groundtruth.csv')
    
    # Download novel if needed
    novel_file = download_novel()
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Create and populate vector database
    with VectorDatabase('eval_vector_store.db') as db:
        # Read and store novel content
        with open(novel_file, 'r', encoding='utf-8') as f:
            content = f.read()
        db.put(novel_file, content)
        
        # Initialize RAG
        rag = RetrievalAugmentedGenerator(db)
        
        # Evaluate each ground truth QA pair
        context_scores = []
        answer_scores = []
        
        for _, row in ground_truth.iterrows():
            question = row['Question']
            ground_truth_answer = row['Answer']
            
            # Generate response using RAG
            result = rag.generate(question)
            
            # Evaluate using LLM
            context_score, answer_score = evaluate_response(
                client,
                result['response'],
                f"Question: {question}\nAnswer: {ground_truth_answer}",
                result['context']
            )
            
            context_scores.append(context_score)
            answer_scores.append(answer_score)
            
            print(f"Question: {question}")
            print(f"Context Score: {context_score:.2f}")
            print(f"Answer Score: {answer_score:.2f}\n")
    
    # Plot score distributions
    plot_score_distributions(context_scores, answer_scores)
    
    # Calculate and print average scores
    print(f"Average Context Score: {sum(context_scores)/len(context_scores):.2f}")
    print(f"Average Answer Score: {sum(answer_scores)/len(answer_scores):.2f}")
    
    # Clean up by removing the vector database
    if os.path.exists('eval_vector_store.db'):
        os.remove('eval_vector_store.db')

if __name__ == "__main__":
    main()
