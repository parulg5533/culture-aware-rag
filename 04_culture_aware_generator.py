import os
import sys
import pandas as pd
import time
from tqdm import tqdm
import importlib

# Import functions from previous script (importlib is used since the file starts with a number)
retriever_module = importlib.import_module("03_culture_aware_retriever")
setup_gemini = retriever_module.setup_gemini
extract_cultural_entities = retriever_module.extract_cultural_entities
retrieve_local_evidence = retriever_module.retrieve_local_evidence

def generate_rag_answer(model, question, evidence_snippets, language="Hindi"):
    """Generates an answer using the LLM and the retrieved evidence."""
    system_instruction = (
        f"You are a helpful and culturally-aware AI assistant. "
        f"Answer the following question accurately in {language}.\n"
        f"Use the provided 'Retrieved Local Evidence' to inform your answer, especially if it contains specific cultural context or facts.\n"
        f"If the evidence is not relevant, rely on your own knowledge."
    )
    
    # Format evidence
    evidence_text = ""
    if evidence_snippets:
        for i, snippet in enumerate(evidence_snippets, 1):
            evidence_text += f"Source {i}: {snippet['title']}\nSnippet: {snippet['snippet']}\n\n"
    else:
        evidence_text = "No additional local evidence found."

    prompt = f"{system_instruction}\n\nRetrieved Local Evidence:\n{evidence_text}\n\nQuestion: {question}\nAnswer:"
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"ERROR: {e}"

def run_generator():
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        model = setup_gemini()
    except Exception as e:
        print(f"Setup Error: {e}")
        return

    # Load the baseline sample so we can add our RAG results directly next to it for comparison
    input_file = os.path.join(os.path.dirname(__file__), "baseline_results_sample.csv")
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Please run 02_run_baseline.py first.")
        return

    print(f"Loading data from {input_file}...")
    df_sample = pd.read_csv(input_file)
    
    rag_answers = []
    retrieved_contexts = []
    
    print(f"Running Culture-Aware Generation on {len(df_sample)} questions...")
    
    for index, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Processing Pipeline"):
        question = row['question']
        language = row.get('language_name', 'Hindi')
        
        # Step 1: Analyze and Retrieve
        analysis = extract_cultural_entities(model, question, language)
        search_query = analysis.get('search_query', question)
        evidence = retrieve_local_evidence(search_query)
        
        # Save context for analysis later
        context_str = " | ".join([f"{e['title']}: {e['snippet'][:100]}" for e in evidence])
        retrieved_contexts.append(context_str)
        
        # Step 2: Generate Answer
        rag_answer = generate_rag_answer(model, question, evidence, language)
        rag_answers.append(rag_answer)
        
        # Simple rate limiting protection
        time.sleep(2)
        
    df_sample['rag_retrieved_context'] = retrieved_contexts
    df_sample['rag_generated_answer'] = rag_answers
    
    output_file = os.path.join(os.path.dirname(__file__), "rag_results_sample.csv")
    df_sample.to_csv(output_file, index=False)
    print(f"\nPipeline complete! RAG results saved to {output_file}")
    
    print("\nSample Comparison:")
    for idx, row in df_sample.head(2).iterrows():
        print("=" * 60)
        print(f"Q: {row['question']}")
        print("-" * 30)
        print(f"Baseline Answer: {str(row['baseline_generated_answer'])[:150]}...")
        print("-" * 30)
        print(f"RAG Context: {str(row['rag_retrieved_context'])[:100]}...")
        print(f"RAG Answer: {str(row['rag_generated_answer'])[:150]}...")
        print("=" * 60)

if __name__ == "__main__":
    run_generator()
