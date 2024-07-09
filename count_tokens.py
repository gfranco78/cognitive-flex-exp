import pandas as pd
import tiktoken
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
PROCESSED_CSV = 'comparison_results.csv'  # Path to the processed results CSV
PROMPT_COLUMNS = ['prompt_A_full_answer', 'prompt_B_full_answer', 'prompt_C_full_answer', 'prompt_D_full_answer', 
                  'prompt_E_full_answer', 'prompt_F_full_answer', 'prompt_G_full_answer', 'prompt_H_full_answer']

# Load the processed CSV
def load_processed_csv(file_path):
    logging.info(f"Loading processed CSV from {file_path}")
    return pd.read_csv(file_path)

# Function to count tokens in a string using tiktoken
def count_tokens(text):
    if pd.isna(text):
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

# Main function to count tokens for each prompt column
def count_tokens_in_prompts(df, prompt_columns):
    for column in prompt_columns:
        logging.info(f"Counting tokens for {column}")
        df[f'{column}_token_count'] = df[column].apply(count_tokens)
    return df

# Load the processed CSV
df = load_processed_csv(PROCESSED_CSV)

# Count tokens in each prompt column
df = count_tokens_in_prompts(df, PROMPT_COLUMNS)

# Display token counts
for column in PROMPT_COLUMNS:
    total_tokens = df[f'{column}_token_count'].sum()
    print(f"Total tokens in {column}: {total_tokens}")

# Export the updated DataFrame with token counts to a new CSV
df.to_csv('comparison_results_with_token_counts.csv', index=False)
print("Updated results have been exported to comparison_results_with_token_counts.csv")