import os
from datasets import load_dataset
import pandas as pd

def main():
    print("Loading CaLMQA dataset from Hugging Face...")
    try:
        # Load the dataset
        # We'll load the full dataset first to see what's available
        dataset = load_dataset("shanearora/CaLMQA", split="train") 
        df = dataset.to_pandas()
        
        print(f"Total rows in dataset: {len(df)}")
        print("\nAvailable columns:")
        print(df.columns.tolist())
        
        # Check what languages are present
        if 'language' in df.columns:
            # Check if it's a ClassLabel to get names
            if hasattr(dataset.features['language'], 'names'):
                lang_names = dataset.features['language'].names
                print("\nLanguage mapping found in features:")
                print(lang_names)
                # Convert ints to strings
                df['language_name'] = df['language'].apply(lambda x: lang_names[x] if x < len(lang_names) else str(x))
            else:
                print("No feature names found, casting to string.")
                df['language_name'] = df['language'].astype(str)
                
            print("\nLanguage distribution:")
            print(df['language_name'].value_counts())
            
            indian_langs = ['hi', 'bn', 'ta', 'te', 'mr', 'ur', 'gu', 'ml', 'kn', 'or', 'pa', 'as']
            indian_langs_names = ['hindi', 'bengali', 'tamil', 'telugu', 'marathi', 'urdu', 'gujarati', 'malayalam', 'kannada', 'odia', 'punjabi', 'assamese']
            
            mask = df['language_name'].str.lower().isin(indian_langs + indian_langs_names)
            indian_df = df[mask]
            
            print(f"\nFiltered Indian language subset size: {len(indian_df)}")
            
            if len(indian_df) > 0:
                print(indian_df['language_name'].value_counts())
                # Save to CSV
                output_path = os.path.join(os.path.dirname(__file__), "calmqa_indian_subset.csv")
                indian_df.to_csv(output_path, index=False)
                print(f"\nSaved subset to {output_path}")
            else:
                print("\nNo Indian languages found using the initial filter list. We may need to adjust the filter criteria.")
                # Save a sample to inspect
                df.head(100).to_csv("calmqa_sample.csv", index=False)
                print("Saved a sample of 100 rows to calmqa_sample.csv for inspection.")
        else:
            print("Could not find a 'language' column. Saving a sample to inspect.")
            df.head(100).to_csv("calmqa_sample.csv", index=False)
            
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()
