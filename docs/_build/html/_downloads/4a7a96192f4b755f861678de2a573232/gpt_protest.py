###############################
# Protest Images - Workshop Ex GPT 
###############################
# Lab 3 Answer

# libraries
import os
import csv
from openai import OpenAI 
from pydantic import BaseModel, ValidationError
from typing import List 
from datetime import datetime
import pandas as pd
import time
import json
import base64

#########
# Inputs
#########

# API Token
api_file = "/home/<your_net_id>/keys/api_key.txt"
with open(api_file, 'r') as file:
    api_key = file.read().strip()
client = OpenAI(api_key=api_key)

# settings
model_select = 'gpt-4o' 
temp_set = 0
top_pset = 0.0
seed_set = 42

# Prompt Text
system_prompt = """
"You are an objective image analyst. Your task is to accurately describe protest scenes and identify visible acts of violence.
"""

user_prompt= """
Task: Analyze the provided image of protest activity to determine if violence or violent activities are occurring.

**Fields to Extract:**
1.  **Image Description (image_desc):**
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

"""

input_dir = "/home/<your_net_id>/extract-book/data/insurrection/"

# outputs
output_dir = "/home/<your_net_id>/extract-book/code/lab3"
results_track = os.path.join(output_dir, "protest_results_track.csv")

#########
# Schema
#########

# New Schema for a single object with two fields
class ProtestResponse(BaseModel):
    image_desc: str
    violence: str
    justification_text: str

#########
# Functions
#########

# encode image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# api call
def call_gpt(user_prompt, system_prompt, schema, model_select, temperature, top_p, seed_value, image_base64):
    try:
        response = client.beta.chat.completions.parse( 
            model=model_select,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]} 
            ],
            temperature=temperature, 
            top_p=top_p,             
            response_format=schema, 
            seed=seed_value         
        )
        
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error calling OpenAI API with get_response: {e}")
        return None

# Create tracker file if doesn't exist
def init_tracker(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Updated header to include the two response fields
            writer.writerow(["timestamp", "prompt", "file_name", "model", "temperature", "top_p", "image_desc", "violence", "justification_text"])
        print(f"Created new tracker at {csv_path}")

# Append results to tracker
def log_response(csv_path, prompt, file_name, model, temperature, top_p, image_desc, violence, justification_text):
    now = datetime.now().isoformat()
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([now, prompt, file_name, model, temperature, top_p, image_desc, violence, justification_text])

# Run all prompts
def run_prompt(prompt, file_name, system_prompt, model_select, temperature, top_p, seed_value, response_schema, csv_path, image_path):

    raw_json_response = call_gpt(prompt, system_prompt, response_schema, model_select, temperature, top_p, seed_value, encode_image(image_path))
    
    # Initialize variables for logging with default values
    image_desc = ""
    violence = ""
    justification_text = ""

    if raw_json_response:
        try:
            # Use the provided schema to parse and validate the JSON response
            parsed_data = response_schema.model_validate_json(raw_json_response)
            
            # Extract fields from the parsed data
            image_desc = parsed_data.image_desc
            violence = parsed_data.violence
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
    log_response(csv_path, prompt, file_name, model_select, temperature, top_p, image_desc, violence, justification_text)

#####
# Run
#####

def main():
    # get a list of all .png files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

     # limit to first 5 emails for testing
    image_files = image_files[:2]
    print("I obtained a list of images to process.")
    init_tracker(results_track)

    for img in image_files:
        file_name = os.path.splitext(img)[0]
        image_file = os.path.join(input_dir, img)
        print(f"Processing image: {image_file}...")
        # Run the prompt
        run_prompt(user_prompt, file_name, system_prompt, model_select, temp_set, top_pset, seed_set, ProtestResponse, results_track, image_file)
        time.sleep(5)    

if __name__ == "__main__":
    main()
