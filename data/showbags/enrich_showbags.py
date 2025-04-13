# enrich_showbags.py

# This script uses a baseten hosted  to enrich the showbags dataset 
## with descriptions of the images.
# It uses the Mistral 3.1 small model.
# It saves the enriched data to a new CSV file.
# It also saves intermediate results every 5 images to avoid losing progress.

import pandas as pd
import os
import time
from openai import OpenAI

def get_image_description(image_url):
    """
    Call the AI model to get a description of the showbag image.
    
    Args:
        image_url (str): URL of the showbag image
        
    Returns:
        str: Description of the image contents
    """
    client = OpenAI(
        api_key=os.getenv("BASETEN_API_KEY"),
        base_url="https://model-zq8dmgpw.api.baseten.co/environments/production/sync/v1"
    )

    try:
        response = client.chat.completions.create(
            model="mistral",
            messages=[{"role": "user",
                     "content": [
                         {"text": "You have an image of a showbag. Describe in a paragraph the sorts of contents, the brands or characters involved, and the type of person this may appeal to.", "type": "text"},
                         {"type": "image_url",
                          "image_url": {"url": image_url}}
                     ]
                    }],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return "Error: Could not generate description"

def main():
    # File paths
    input_file = 'showbags.csv'
    output_file = 'showbags_enriched.csv'
    
    # Check if the enriched file already exists
    if os.path.exists(output_file):
        print(f"Found existing {output_file}, will resume processing...")
        df = pd.read_csv(output_file)
    else:
        print(f"Loading {input_file}...")
        df = pd.read_csv(input_file)
        df['img_description'] = None
    
    # Count how many rows need processing
    rows_to_process = df[pd.isna(df['img_description'])].index
    total_to_process = len(rows_to_process)
    
    if total_to_process == 0:
        print("All images already have descriptions. Nothing to do.")
        return
    
    print(f"Processing {total_to_process} showbags that need descriptions...")
    
    # Process each row that doesn't have a description
    for idx, i in enumerate(rows_to_process):
        if idx % 5 == 0:
            print(f"Processing showbag {idx+1}/{total_to_process}")
            # Save intermediate results every 5 images
            if idx > 0: 
                df.to_csv(output_file, index=False)
                print(f"Saved progress to {output_file}")
        
        # Get the image description
        image_url = df.loc[i, 'image_url']
        if pd.notna(image_url) and image_url.strip():
            description = get_image_description(image_url)
            df.at[i, 'img_description'] = description
            
            # Add a short delay to avoid overloading the API
            time.sleep(1)
    
    # Save the final enriched dataframe
    print(f"Saving enriched data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    main()

