import json
import os

def generate_wlasl_labels():
    # Make sure WLASL_v0.3.json is in the same folder as this script
    dataset_file = "WLASL_v0.3.json"
    
    # We will save it directly to your project folder (update this path if needed)
    output_file = "labels.json"

    print(f"Reading {dataset_file}...")
    try:
        with open(dataset_file, 'r') as f:
            content = json.load(f)
    except FileNotFoundError:
        print(f"❌ Could not find {dataset_file}. Please put this script in the same folder as WLASL_v0.3.json")
        return

    # Extract all the words and sort them alphabetically (this perfectly matches 
    # the LabelEncoder used by the researchers during training)
    words = sorted([entry['gloss'] for entry in content])

    print(f"Found {len(words)} unique words.")

    # Save to JSON array format
    with open(output_file, 'w') as f:
        json.dump(words, f, indent=4)

    print(f"✅ Successfully created {output_file}!")
    print(f"First 5 words: {words[:5]}")
    print(f"Last 5 words: {words[-5:]}")

if __name__ == "__main__":
    generate_wlasl_labels()