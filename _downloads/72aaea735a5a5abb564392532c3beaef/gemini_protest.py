###############################
# Protest Images - Workshop Ex Gemini
###############################
# Lab 3 Answer (Clean, Corrected)

# libraries
import os
import csv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError
from datetime import datetime
import time

#########
# Inputs
#########

# API Token
api_file = "/home/<your_net_id>/keys/gemini-key.txt"
with open(api_file, 'r') as file:
    api_key = file.read().strip()
client = genai.Client(api_key=api_key)

# settings
model_select = 'gemini-2.0-flash'
temp_set = 0.0
top_pset = 0.0

# Prompt Text
user_prompt = """
Task: Analyze the provided image of protest activity to determine if violence or violent activities are occurring.

**Fields to Extract:**
1.  **Image Description (image_desc):**
    * A concise phrase (typically 2-5 words) summarizing the most prominent action or subject depicted in the image.
2.  **Violence (violence):**
    * "yes" or "no" based on observed violence.
3.  **Justification for Violence Assessment (justification_text):**
    * Specific, observable visual cues that directly support your assessment.

**Output Structure:**
Your response must be a JSON object matching this structure:
{
  "image_desc": "...",
  "violence": "...",
  "justification_text": "..."
}
"""

input_dir = "/home/<your_net_id>/extract-book/data/insurrection/"
output_dir = "/home/<your_net_id>/extract-book/code/lab3"
results_track = os.path.join(output_dir, "protest_results_track.csv")

#########
# Schema
#########

class ProtestResponse(BaseModel):
    image_desc: str
    violence: str
    justification_text: str

#########
# Functions
#########


def call_gem(image_path, prompt_text, model_image, DetectedObjectsList, temperature, top_p):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    try:
        response = client.models.generate_content(
            model=model_image,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',  # adjust to 'image/png' if needed
                ),
                prompt_text
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": DetectedObjectsList,
                "temperature": temperature,
                "top_p": top_p
            }
        )

        json_output = response.text
        return json_output

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def init_tracker(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "prompt", "file_name", "model", "temperature", "top_p",
                             "image_desc", "violence", "justification_text"])
        print(f"Created new tracker at {csv_path}")

def log_response(csv_path, prompt, file_name, model, temperature, top_p,
                 image_desc, violence, justification_text):
    now = datetime.now().isoformat()
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([now, prompt, file_name, model, temperature, top_p,
                         image_desc, violence, justification_text])

def run_prompt(prompt, file_name, model_select, temperature, top_p, response_schema, csv_path, image_path):
    raw_json_response = call_gem(image_path, prompt, model_select, response_schema, temperature, top_p)

    # Initialize defaults
    image_desc = ""
    violence = ""
    justification_text = ""

    if raw_json_response:
        try:
            parsed_data = response_schema.model_validate_json(raw_json_response)
            image_desc = parsed_data.image_desc
            violence = parsed_data.violence
            justification_text = parsed_data.justification_text

        except ValidationError as e:
            print(f"Schema validation error for image '{file_name}': {e}")
            print(f"Raw JSON: {raw_json_response}")
        except Exception as e:
            print(f"Unexpected parsing error for image '{file_name}': {e}")
            print(f"Raw JSON: {raw_json_response}")
    else:
        print(f"No response received for image '{file_name}'.")

    log_response(csv_path, prompt, file_name, model_select, temperature, top_p,
                 image_desc, violence, justification_text)

#####
# Run
#####

def main():
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    image_files = image_files[:2]  # Limit to first 2 for testing

    print(f"Found {len(image_files)} image(s) to process.")
    init_tracker(results_track)

    for img in image_files:
        file_name = os.path.splitext(img)[0]
        image_path = os.path.join(input_dir, img)
        print(f"Processing: {image_path}")
        run_prompt(user_prompt, file_name, model_select, temp_set, top_pset, ProtestResponse, results_track, image_path)
        time.sleep(5)

if __name__ == "__main__":
    main()

