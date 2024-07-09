import pandas as pd
import ast
import re
import os
import logging
import time
from openai import OpenAI


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Initialize OpenAI client
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()

# Constants
SAMPLES_PER_CATEGORY = 10  # Adjust this number as needed to sample from the 14 categories
NUM_QUESTIONS_TO_TEST = 140  # Adjust this number as needed for testing
PROMPT_TYPES = ['A','B','C','D','E','F','G','H']  # Control over list of prompt templates to process
DELAY_BETWEEN_REQUESTS = 2  # Delay in seconds between processing each question

# Load the dataset
def load_dataset(file_path):
    logging.info(f"Loading dataset from {file_path}")
    return pd.read_parquet(file_path)

# Function to get random samples from each group
def get_random_samples(group, n):
    return group.sample(n=n, random_state=101)

# Function to format answer options
def format_options(options):
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    if isinstance(options, str):
        options = ast.literal_eval(options)
    formatted_options = [f"{label}: {option.strip()}" for label, option in zip(option_labels, options)]
    return '\n'.join(formatted_options)

# Function to create prompts based on parts
def create_prompt(row, part2, part4):
    prompt = (
        f"{row['question']}\n\n"
        f"{part2}\n\n"
        f"{row['formatted_options']}\n\n"
        f"{part4}\n\n"
    )
    return prompt

# Function to get GPT answer
def get_gpt_answer(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a knowledgeable model skilled at answering multiple choice questions."},
                {"role": "user", "content": prompt}
            ]
        )
        full_answer = response.choices[0].message.content.strip()
        extracted_answer = next((word[0] for word in full_answer.split() if re.match(r'^[A-J]:', word)), "")
        return full_answer, extracted_answer
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return None, None

# Main function to process the dataset and generate results
def process_dataset(df, num_samples, num_questions, prompt_types):
    # Sample the data
    sampled_df_list = [get_random_samples(group, num_samples) for _, group in df.groupby('category')]
    sampled_df = pd.concat(sampled_df_list).reset_index(drop=True)

    # Apply the function to create formatted options
    sampled_df['formatted_options'] = sampled_df['options'].apply(format_options)

    # Prompt configuration
    prompt_config = {
        'A': ("Select the correct answer from the following answer choices:", 
              "Respond with the letter and answer selected."),
        'B': ("Select the correct answer from the following answer choices:", 
              "Respond only with the letter and answer selected and nothing else."),
        'C': ("Let's think step by step and select the correct answer from the following answer choices:", 
              "Respond with the letter and answer selected."),
        'D': ("Let's think step by step and select the correct answer from the following answer choices:", 
              "Respond only with the letter and answer selected and nothing else."),
        'E': ("Let's think about the knowledge and concepts needed and select the correct answer from the following answer choices:", 
              "Respond with the letter and answer selected."),  
        'F': ("Let's think about the knowledge and concepts needed and select the correct answer from the following answer choices:", 
              "Respond only with the letter and answer selected and nothing else."),
        'G': ("My expectations are that you will answer the question correctly. Create an operational context for yourself to maximize"
              "fullfillment of my expectations and select the correct answer from the following answer choices:", 
              "Respond with the letter and answer selected."),
        'H': ("My expectations are that you will answer the question correctly. Create an operational context for yourself to maximize"
              "fullfillment of my expectations and select the correct answer from the following answer choices:",  
              "Respond only with the letter and answer selected and nothing else.")
    }

    # Apply the function to create prompt columns
    for prompt_type in prompt_types:
        part2, part4 = prompt_config[prompt_type]
        sampled_df[f'prompt_{prompt_type}'] = sampled_df.apply(lambda row: create_prompt(row, part2, part4), axis=1)

    # Initialize the results list
    results = []

    # Checking answers for the examples
    for idx, row in sampled_df.head(num_questions).iterrows():
        logging.info(f"Processing question {idx + 1}/{num_questions} (Question ID: {row['question_id']})")
        correct_answer = row['answer']
        formatted_options = row['formatted_options']

        result = {'question_id': row['question_id'], 'category': row['category'], 'question': row['question'],
                  'options': formatted_options, 'correct_answer': correct_answer}

        for prompt_type in prompt_types:
            full_answer, extracted_answer = get_gpt_answer(row[f'prompt_{prompt_type}'])
            if extracted_answer is None:
                logging.warning(f"Question {idx + 1} (Question ID: {row['question_id']}) failed for prompt {prompt_type}")
                break  # Skip this question if any prompt resulted in an error
            result[f'prompt_{prompt_type}_extracted_answer'] = extracted_answer
            result[f'prompt_{prompt_type}_full_answer'] = full_answer

        if all(result.get(f'prompt_{pt}_extracted_answer') is not None for pt in prompt_types):
            results.append(result)
            logging.info(f"Question {idx + 1} (Question ID: {row['question_id']}) processed successfully")
        else:
            logging.warning(f"Question {idx + 1} (Question ID: {row['question_id']}) skipped due to errors in processing")
                # Add a delay between requests
        time.sleep(DELAY_BETWEEN_REQUESTS)

    return pd.DataFrame(results)

# Define dataset splits
splits = {
    'test': 'data/test-00000-of-00001.parquet',
    'validation': 'data/validation-00000-of-00001.parquet'
}

# Load the dataset
df = load_dataset("hf://datasets/TIGER-Lab/MMLU-Pro/" + splits["test"])

# Process the dataset with selected prompt types
results_df = process_dataset(df, SAMPLES_PER_CATEGORY, NUM_QUESTIONS_TO_TEST, PROMPT_TYPES)

# Display the results DataFrame
print(results_df)

# Calculate and print accuracy for each prompt type
for prompt_type in PROMPT_TYPES:
    accuracy = (results_df['correct_answer'] == results_df[f'prompt_{prompt_type}_extracted_answer']).mean() * 100
    print(f"Accuracy for Prompt {prompt_type}: {accuracy:.2f}%")

# Export results to CSV
results_df.to_csv('comparison_results.csv', index=False)
print("Results have been exported to comparison_results.csv")