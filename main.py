import argparse
from scripts.vectordb import VectorDatabase
import os
from scripts.rag import RetrievalAugmentedGenerator

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Search a document using vector database')
    parser.add_argument('filepath', type=str, help='Path to the text file to search')
    parser.add_argument('question', type=str, help='Search query/question')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.filepath):
        print(f"Error: File {args.filepath} does not exist")
        return
    
    # Read the file
    try:
        with open(args.filepath, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Initialize vector database and perform search
    with VectorDatabase() as db:
        # Store the document
        filename = os.path.basename(args.filepath)
        db.put(filename, content)
        
        # Initialize the RAG system and generate answer
        rag = RetrievalAugmentedGenerator(db)
        result = rag.generate(args.question)
        
        # Write results to analysis.txt
        with open('analysis.txt', 'w', encoding='utf-8') as outfile:
            outfile.write("Question:\n")
            outfile.write(args.question)
            outfile.write("\n\nAnswer:\n")
            outfile.write(result['response'])
            outfile.write("\n\nEvidence:\n")
            outfile.write(result['context'])
            outfile.write("\n")

if __name__ == "__main__":
    main()
