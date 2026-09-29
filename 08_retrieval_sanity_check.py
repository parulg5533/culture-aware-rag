import os
import sys
import json
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

def setup_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file.")
        sys.exit(1)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-3.5-flash-lite')

def get_query_strategies(model, question):
    prompt = f"""
    You are a query formulation expert. Given the following question in Hindi, generate three specific search query strategies.
    
    Question: "{question}"
    
    Respond ONLY with a valid JSON object matching this schema:
    {{
        "B_llm_hindi": "A highly specific 3-5 word search query in Hindi",
        "C_english": "The direct English translation of the most effective search terms (3-5 words)",
        "D_hybrid": "A mix of the core Hindi entity name and English keywords (e.g., 'अर्जुन छाल medicinal uses')"
    }}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        print(f"Error generating strategies: {e}")
        return {"B_llm_hindi": question, "C_english": "error", "D_hybrid": "error"}

def execute_search(query, region):
    start_time = time.time()
    result_data = {
        "status": "UNKNOWN",
        "error_type": "None",
        "num_results": 0,
        "latency_ms": 0,
        "results": []
    }
    
    try:
        with DDGS() as ddgs:
            # backend='api' is usually most stable, but we use default 'auto' to mimic baseline
            results = ddgs.text(query, region=region, max_results=3, backend='auto')
            
            # Note: ddgs.text returns a list of dicts or a generator
            res_list = list(results) if results else []
            
            result_data["num_results"] = len(res_list)
            if len(res_list) > 0:
                result_data["status"] = "SUCCESS_WITH_RESULTS"
                result_data["results"] = res_list
            else:
                result_data["status"] = "SUCCESS_ZERO_RESULTS"
                
    except DuckDuckGoSearchException as e:
        err_str = str(e).lower()
        if "ratelimit" in err_str or "202" in err_str or "429" in err_str:
            result_data["status"] = "RATE_LIMIT"
        elif "timeout" in err_str:
            result_data["status"] = "TIMEOUT"
        else:
            result_data["status"] = "OTHER_EXCEPTION"
        result_data["error_type"] = str(e)
    except Exception as e:
        result_data["status"] = "OTHER_EXCEPTION"
        result_data["error_type"] = str(e)
        
    result_data["latency_ms"] = int((time.time() - start_time) * 1000)
    return result_data

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("Initializing Controlled Retrieval Diagnostic Experiment...")
    
    model = setup_gemini()
    df = pd.read_csv("rag_results_sample.csv")
    print(f"Testing diagnostic queries on {len(df)} questions...\n")
    
    experiment_logs = []
    
    for idx, row in df.iterrows():
        question = row['question']
        print(f"--- Q{idx+1}: {question} ---")
        
        # Formulate Queries
        strategies = get_query_strategies(model, question)
        queries = {
            "A": question,
            "B": strategies.get("B_llm_hindi", question),
            "C": strategies.get("C_english", ""),
            "D": strategies.get("D_hybrid", "")
        }
        
        # Test configurations
        configs = [
            ("A", "us-en"), ("A", "in-en"),
            ("B", "us-en"), ("B", "in-en"),
            ("C", "in-en"),
            ("D", "in-en")
        ]
        
        for strat, region in configs:
            query = queries[strat]
            if not query or query == "error": continue
                
            print(f"  Testing [{strat}] in [{region}]: '{query}'")
            res = execute_search(query, region)
            
            log_entry = {
                "question": question,
                "strategy": strat,
                "query": query,
                "region": region,
                "backend": "auto",
                "status": res["status"],
                "error_type": res["error_type"],
                "latency_ms": res["latency_ms"],
                "num_results": res["num_results"],
                "result_1_title": "", "result_1_url": "", "result_1_snippet": "",
                "result_2_title": "", "result_2_url": "", "result_2_snippet": "",
                "result_3_title": "", "result_3_url": "", "result_3_snippet": ""
            }
            
            for i, r in enumerate(res["results"]):
                if i >= 3: break
                log_entry[f"result_{i+1}_title"] = r.get("title", "")
                log_entry[f"result_{i+1}_url"] = r.get("href", "")
                log_entry[f"result_{i+1}_snippet"] = r.get("body", "")
                
            experiment_logs.append(log_entry)
            print(f"    -> {res['status']} ({res['num_results']} results) in {res['latency_ms']}ms")
            
            time.sleep(2) # VERY IMPORTANT: 2 second delay to avoid instant rate limiting
            
    log_df = pd.DataFrame(experiment_logs)
    log_df.to_csv("retrieval_diagnostic_logs.csv", index=False, encoding='utf-8')
    print("\nDiagnostic complete. Saved to retrieval_diagnostic_logs.csv")

if __name__ == "__main__":
    main()
