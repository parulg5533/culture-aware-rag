import os
import sys
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import time

def setup_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("Please set your GEMINI_API_KEY in the .env file")
    
    genai.configure(api_key=api_key)
    # Using gemini-3.5-flash-lite to avoid rate limits
    model = genai.GenerativeModel('gemini-3.5-flash-lite')
    return model

def extract_cultural_entities(model, question, language="Hindi"):
    """Uses LLM to analyze the question and extract cultural entities and region context."""
    prompt = f"""
    You are a cultural linguistics expert. Analyze the following question in {language}.
    Your task is to identify the core cultural entities, traditions, places, or concepts that someone would need local knowledge about to answer the question correctly.
    
    Question: "{question}"
    
    Respond ONLY with a valid JSON object matching this schema:
    {{
        "cultural_entities": ["entity1", "entity2"],
        "region_context": "The specific region or state in India if implied, otherwise just 'India'",
        "search_query": "A highly specific 3-5 word search query in Hindi to look up this cultural fact"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean up markdown JSON block if present
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        result = json.loads(text.strip())
        return result
    except Exception as e:
        print(f"Error during entity extraction: {e}")
        # Fallback
        return {
            "cultural_entities": [],
            "region_context": "India",
            "search_query": question[:50] # truncated question as fallback
        }

def retrieve_local_evidence(search_query):
    """Uses DuckDuckGo to search for the query and returns the top snippets."""
    print(f"  -> Performing search for: '{search_query}'")
    snippets = []
    try:
        # DDGS allows us to search without an API key
        with DDGS() as ddgs:
            # The LLM query is already in Hindi, so we don't need to append English words
            # We can just search directly to get better local results
            results = ddgs.text(search_query, max_results=3)
            for r in results:
                snippets.append({
                    "title": r.get('title', ''),
                    "snippet": r.get('body', ''),
                    "link": r.get('href', '')
                })
    except Exception as e:
        print(f"  -> Search failed: {e}")
        
    # Rate limiting for public API
    time.sleep(2)
    return snippets

def test_retriever():
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        model = setup_gemini()
    except Exception as e:
        print(f"Setup Error: {e}")
        return

    input_file = os.path.join(os.path.dirname(__file__), "calmqa_indian_subset.csv")
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    
    # Let's test on 3 sample questions
    print("Testing Culture-Aware Retriever Pipeline on 3 sample questions...\n")
    sample_df = df.sample(n=3, random_state=42)
    
    for idx, row in sample_df.iterrows():
        question = row['question']
        print("="*60)
        print(f"QUESTION: {question}")
        print("="*60)
        
        # Step 1: Analyze
        print("1. Extracting Cultural Context...")
        analysis = extract_cultural_entities(model, question)
        print(f"   Entities: {analysis.get('cultural_entities')}")
        print(f"   Region: {analysis.get('region_context')}")
        print(f"   Formulated Query: {analysis.get('search_query')}")
        
        # Step 2: Retrieve
        print("\n2. Retrieving Local Evidence...")
        evidence = retrieve_local_evidence(analysis.get('search_query'))
        
        if evidence:
            for i, ev in enumerate(evidence, 1):
                print(f"   [Source {i}] {ev['title']}")
                print(f"   Snippet: {ev['snippet'][:150]}...")
        else:
            print("   No evidence found.")
            
        print("\n")

if __name__ == "__main__":
    test_retriever()
