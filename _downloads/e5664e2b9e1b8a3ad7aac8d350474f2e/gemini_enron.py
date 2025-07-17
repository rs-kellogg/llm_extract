###############################
# Enron Emails - Workshop Ex GPT 
###############################
# Lab 2 Answer

# libraries
import os
import csv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Literal
from pydantic import Field
from datetime import datetime
import pandas as pd
import time
import json


#########
# Inputs
#########

# API Token
api_file = "/home/<your_net_id>/keys/gemini-key.txt"
with open(api_file, 'r') as file:
    api_key = file.read().strip()
client = genai.Client(api_key=api_key)

# settings
model_select = 'gemini-1.5-flash' 
temp_set = 0
top_pset = 0.0
# Directories
enron_data_dir = "/home/<your_net_id>/extract-book/data/enron_emails2/" 
output_base_dir = "/home/<your_net_id>/extract-book/code/lab2/"

os.makedirs(output_base_dir, exist_ok=True)

results_track = os.path.join(output_base_dir, "enron_email_extractions.csv")


#########
# Prompts
#########

user_prompt_protest_violence_detection = """
Task: Analyze the provided image of protest activity to determine if violence or violent activities are occurring.

**Fields to Extract:**
1.  **Image Description (image_description):**
    * A concise phrase (typically 2-5 words) summarizing the most prominent action or subject depicted in the image.
    * Examples: "Crowd marching with signs", "Police engaging protesters", "Vehicle on fire during protest".
2.  **Violence (violence):**
    * **"yes"**: If the image clearly depicts physical aggression (e.g., fighting, throwing projectiles, assaults, use of weapons), active destruction of property (e.g., smashing, burning, vandalism), or direct, forceful clashes between groups or individuals.
    * **"no"**: If the image shows only peaceful demonstrations, orderly gatherings, or passive resistance without any visible acts of physical harm, aggression, or property destruction.
3.  **Justification for Violence Assessment (justification_text):**
    * Provide specific, observable visual cues and details from the image that directly support your 'violence' assessment.
    * If "yes", describe the visible violent acts or property damage (e.g., "Protesters throwing bottles at police line," "A storefront window is shattered," "Smoke rising from a burning car," "Physical altercation between two individuals").
    * If "no", explain what makes the scene appear peaceful (e.g., "Participants are holding banners and chanting peacefully," "No signs of aggression or damage," "People are sitting calmly in protest").

**Output Structure:**
Your response must be a JSON object conforming to the provided schema.

Here is the image content to analyze:
(Note: The image content itself will be provided directly to the multimodal model API alongside this text prompt.)
"""


#########
# Schema
#########
# New Schema for a single object with two fields
class EmailExtraction(BaseModel):
    to_recipients: List[str] = Field(description="List of primary recipients in the 'To:' field.")
    from_sender: str = Field(description="Sender's email address or name from the 'From:' field.")
    email_date: Optional[str] = Field(None, description="Date the email was sent, in YYYY-MM-DD format.")
    subject: str = Field(description="Subject line of the email.")
    nefarious_activity_flag: Literal["yes", "no", "uncertain"] = Field(
        description="Assessment of whether the email discusses or implies nefarious or illegal activity."
    )
    justification_text: str = Field(
        description="Specific text or summary from the email that justifies the nefarious activity assessment."
    )


#########
# Functions
#########

def get_content(file_path: str) -> Optional[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def call_gem(prompt_text, model_select, temp_set, top_pset, response_schema):
    try:
        response = client.models.generate_content(
            model=model_select,
            contents=[
                prompt_text
            ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                    "temperature": temp_set,
                    "top_p": top_pset
                }
        )

        json_output = response.text
        return json_output

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

def init_tracker(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Updated header to include the two response fields
            writer.writerow(["timestamp", "prompt", "model", "temperature", "top_p", "to_recipients", "from_sender", "email_date", "subject", "nefarious_activity_flag", "justification_text"])
        # Create the tracker file with the correct header
        print(f"Created new tracker at {csv_path}")

# Append results to tracker
def log_response(csv_path, prompt, model, temperature, top_p, to_recipients, from_sender, email_date, subject, nefarious_activity_flag, justification_text):
    now = datetime.now().isoformat()
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([now, prompt, model, temperature, top_p, 
                         to_recipients, from_sender, email_date, subject, 
                         nefarious_activity_flag, justification_text])

# Run all prompts
def run_prompt(prompt, model_select, temperature, top_p, response_schema, csv_path):
    print(f"Processing prompt: {prompt[:100]}...") # Print first 100 chars to avoid very long print

    raw_json_response = call_gem(prompt, model_select, temp_set, top_pset, response_schema)
    
    # Initialize variables for logging with default values
    to_recipients = []
    from_sender = ""
    email_date = None # Consider using datetime.date or str depending on schema
    subject = ""
    nefarious_activity_flag = ""
    justification_text = ""

    if raw_json_response:
        try:
            # Use the provided schema to parse and validate the JSON response
            parsed_data = response_schema.model_validate_json(raw_json_response)
            
            # Extract fields from the parsed data
            to_recipients = parsed_data.to_recipients
            from_sender = parsed_data.from_sender
            email_date = parsed_data.email_date
            subject = parsed_data.subject
            nefarious_activity_flag = parsed_data.nefarious_activity_flag
            justification_text = parsed_data.justification_text
            
        except ValidationError as e:
            print(f"Schema validation error for prompt (excerpt: '{prompt[:50]}...'): {e}")
            print(f"Raw JSON response: {raw_json_response}")
        except json.JSONDecodeError as e:
            print(f"JSON decoding error for prompt (excerpt: '{prompt[:50]}...'): {e}")
            print(f"Raw response: {raw_json_response}")
        except Exception as e:
            print(f"An unexpected error occurred during parsing for prompt (excerpt: '{prompt[:50]}...'): {e}")
            print(f"Raw response: {raw_json_response}")
    else:
        print(f"No response or error from call_gpt for prompt (excerpt: '{prompt[:50]}...').")

    # Pass the extracted fields to log_response
    # Note: ensure log_response handles potential empty or default values if parsing failed
    log_response(
        csv_path, prompt, model_select, temperature, top_p,
        to_recipients, from_sender, email_date, subject,
        nefarious_activity_flag, justification_text
    )

def _is_ascii_text_file(filepath: str) -> bool:
    try:
        with open(filepath, 'r', encoding='ascii') as f:
            f.read() # Attempt to read the entire file
        return True
    except (UnicodeDecodeError, Exception):
        # If decoding fails or any other error occurs, it's not a pure ASCII text file
        return False

def find_all_ascii_text_files_minimal(search_directory: str) -> List[str]:
    if not os.path.isdir(search_directory):
        print(f"Error: Directory not found at '{search_directory}'")
        return []

    return [
        os.path.join(root, file_name)
        for root, _, files in os.walk(search_directory)
        for file_name in files
        if _is_ascii_text_file(os.path.join(root, file_name)) # Filter using the external helper
    ]



#########
# Run
#########

def main():
    email_list = find_all_ascii_text_files_minimal(enron_data_dir)

    # limit to first 5 emails for testing
    email_list = email_list[:5]
    print("I obtained a list of emails to process.")
    init_tracker(results_track)

    for email_file in email_list:
        email_content = get_content(email_file)
        if email_content:
            # Prepare the prompt
            user_prompt = f"{user_prompt_enron_extraction}\n\n{email_content}"
            # Run the prompt
            run_prompt(user_prompt, model_select, temp_set, top_pset, EmailExtraction, results_track)
        else:
            print(f"Skipping file {email_file} due to read error or empty content.")

if __name__ == "__main__":
    main()