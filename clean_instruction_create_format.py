import json
import re
from collections import OrderedDict

import re

def extract_objects(symbolic_mod):
    """
    Split only when '+' or '-' is at the start of the line or follows a whitespace character,
    avoiding splitting on hyphens within words like 'flat-screen'.
    """
    # Use (?:^|(?<=\s)) to ensure splitting only when + or - appear at the beginning or following whitespace
    modifications = re.split(r'(?:^|(?<=\s))(?=[+\-])', symbolic_mod.strip())
    modifications = [m.strip() for m in modifications if m.strip()]
    
    added = []
    for mod in modifications:
        if mod.startswith('+'):
            # Remove the leading + and strip surrounding whitespace
            obj = mod[1:].strip()
            added.append(obj)
    
    return ', '.join(added)

def remove_duplicates(conversations):
    """
    Deduplication function: remove duplicates based on messages and images for each record,
    preserve order, and keep only the first occurrence of each duplicate.
    """
    seen = set()
    unique_conversations = []
    for convo in conversations:
        # Generate a unique identifier (considering message content and image paths)
        identifier = json.dumps({
            "messages": sorted(convo["messages"], key=lambda x: x["content"]),
            "images": sorted(convo["images"])
        }, sort_keys=True)
        
        if identifier not in seen:
            seen.add(identifier)
            unique_conversations.append(convo)
    return unique_conversations

def generate_conversation(item):
    """
    Given a data item, return:
    - the original image conversation
    - the modified image conversation (if output_image is valid)
    """
    conversations = []
    
    # Original image conversation
    original_convo = {
        "messages": [
            {"content": "<image>What part of the image has been modified?", "role": "user"},
            {"content": "Nothing has been modified in this image.", "role": "assistant"}
        ],
        "images": [item["image_path"]]
    }
    conversations.append(original_convo)
    
    # If output_image is available, generate the corresponding conversation
    if "Error" not in item["output_path"]:
        symbolic_mod = item["symbolic_modification"]
        objects_str = extract_objects(symbolic_mod)
        modified_content = (
            f"The {objects_str} in the image has been modified."
            if objects_str else
            "Some objects have been modified."
        )
        modified_convo = {
            "messages": [
                {"content": "<image>What part of the image has been modified?", "role": "user"},
                {"content": modified_content, "role": "assistant"}
            ],
            "images": [item["output_path"]]
        }
        conversations.append(modified_convo)
    
    return conversations

def main():
    input_path = '/fdata/FragFake/FragFake_github/Instruction/Gemini-IG/Gemini-IG_easy_instruction_result.json'
    train_output_path = '/fdata/FragFake/FragFake_github/VLM_Dataset/Gemini-IG/easy/Gemini-IG_easy_train_conversation.json'
    test_output_path = '/fdata/FragFake/FragFake_github/VLM_Dataset/Gemini-IG/easy/Gemini-IG_easy_test_conversation.json'

    with open(input_path, 'r') as f:
        original_data = json.load(f)

    # 1. First, find all original image paths (image_path) where test=1
    test_list = set()
    for item in original_data:
        if item.get("test", 0) == 1:
            test_list.add(item["image_path"])

    # 2. Generate conversations for the training set and test set separately
    train_conversations = []
    test_conversations = []

    for item in original_data:
        # Only process specific goals
        if item["modification_goal"] not in ("object addition", "object replacement"):
            continue
        
        # If this item is marked as test=1
        if item.get("test", 0) == 1:
            # Add to test set
            test_conversations.extend(generate_conversation(item))
        else:
            # If image_path is already in test_list, it means the original image is assigned to the test set
            # => discard this item, do not include it in the train or test set
            if item["image_path"] in test_list:
                continue
            else:
                # Otherwise, add it to the training set
                train_conversations.extend(generate_conversation(item))

    # 3. Deduplicate
    train_conversations = remove_duplicates(train_conversations)
    test_conversations = remove_duplicates(test_conversations)

    # Function to count original vs modified image conversations
    def count_images(conversations):
        original_count = 0
        modified_count = 0
        for convo in conversations:
            # Determine whether the assistant message contains "Nothing has been modified"
            # use this to differentiate original image conversations from modified ones
            if any("Nothing has been modified" in m["content"] for m in convo["messages"] if m["role"] == "assistant"):
                original_count += 1
            else:
                modified_count += 1
        return original_count, modified_count

    train_original_count, train_modified_count = count_images(train_conversations)
    test_original_count, test_modified_count = count_images(test_conversations)

    print("Training set:")
    print(f"  Original images: {train_original_count}")
    print(f"  Modified images: {train_modified_count}")
    print(f"  Total conversations: {len(train_conversations)}")
    print("Test set:")
    print(f"  Original images: {test_original_count}")
    print(f"  Modified images: {test_modified_count}")
    print(f"  Total conversations: {len(test_conversations)}")

    # 5. Save to JSON
    with open(train_output_path, 'w') as f:
        json.dump(train_conversations, f, indent=4)

    with open(test_output_path, 'w') as f:
        json.dump(test_conversations, f, indent=4)

if __name__ == "__main__":
    main()
