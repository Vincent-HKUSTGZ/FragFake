import os
import re
import json
import base64
import requests
from tqdm import tqdm
import argparse

# ------------------------------
# Helper Functions
# ------------------------------
def encode_image(image_path):
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_gpt_response(text):
    """Strip Markdown code fences from the GPT response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_added_object(symbolic_modification):
    """
    Extract the added object name from a string like '-foo +bar'.
    Returns 'bar', or None if no '+' is present.
    """
    match = re.search(r"\+(\S.*)$", symbolic_modification)
    return match.group(1).strip() if match else None


def generate_modification_instructions(
    api_key,
    api_url,
    image_path,
    coco_label,
    modification_goal,
    temperature=1.0,
    exclude_object=None
):
    """
    Call GPT-4o Vision to generate:
      - object_caption
      - brief_modification_instruction
      - descriptive_modification
      - symbolic_modification

    If exclude_object is provided, adds a constraint to avoid that object.
    """
    exclude_clause = (
        f"\n\nImportant: Please do NOT use the following object: {exclude_object}\n"
        if exclude_object else ""
    )

    prompt = f"""### Task: Advanced Image Modification Instruction Generation

**Description:**
Given a COCO object label in the image and a modification goal
(e.g., "object addition" or "object replacement"), produce valid JSON
with these fields:
1. object_caption
2. brief_modification_instruction
3. descriptive_modification
4. symbolic_modification
{exclude_clause}
---
**Input:**
- COCO Object Label: {coco_label}
- Modification Goal: {modification_goal}

**Output JSON:**
{{
  "object_caption": "",
  "brief_modification_instruction": "",
  "descriptive_modification": "",
  "symbolic_modification": ""
}}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    img_b64 = encode_image(image_path)
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    }
                ]
            }
        ],
        "temperature": temperature,
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        raw = data["choices"][0]["message"]["content"]
        cleaned = clean_gpt_response(raw)
        parsed = json.loads(cleaned)

        expected_keys = {
            "object_caption",
            "brief_modification_instruction",
            "descriptive_modification",
            "symbolic_modification",
        }
        if expected_keys.issubset(parsed):
            return {k: parsed[k] for k in expected_keys}
        else:
            return {
                "object_caption": "[Error] Missing keys in JSON.",
                "brief_modification_instruction": cleaned,
                "descriptive_modification": "",
                "symbolic_modification": ""
            }

    except requests.HTTPError as e:
        return {
            "object_caption": f"[Error] HTTP error: {e}",
            "brief_modification_instruction": "",
            "descriptive_modification": "",
            "symbolic_modification": ""
        }
    except (json.JSONDecodeError, KeyError) as e:
        return {
            "object_caption": "[Error] Invalid JSON from API.",
            "brief_modification_instruction": str(e),
            "descriptive_modification": "",
            "symbolic_modification": ""
        }
    except Exception as e:
        return {
            "object_caption": f"[Error] Unexpected exception: {e}",
            "brief_modification_instruction": "",
            "descriptive_modification": "",
            "symbolic_modification": ""
        }


# ------------------------------
# Main Processing
# ------------------------------
def first_pass(args):
    """First pass: generate instructions for each image + goal."""
    processed = set()
    results = []

    # Load previous results (skip ones with errors)
    if os.path.isfile(args.output_json):
        try:
            with open(args.output_json, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for item in existing:
                if "[Error]" not in item.get("object_caption", ""):
                    processed.add((item["image_path"], item["modification_goal"]))
                    results.append(item)
        except json.JSONDecodeError:
            print(f"[Warning] Could not parse {args.output_json}, starting fresh.")

    # Walk through image directories
    for category in tqdm(os.listdir(args.base_dir), desc="Categories"):
        category_dir = os.path.join(args.base_dir, category)
        if not os.path.isdir(category_dir):
            continue

        for fname in tqdm(os.listdir(category_dir), desc=f"Images in {category}", leave=False):
            img_path = os.path.join(category_dir, fname)
            if not os.path.isfile(img_path):
                continue

            for goal in args.modification_goals:
                key = (img_path, goal)
                if key in processed:
                    continue

                instr = generate_modification_instructions(
                    api_key=args.api_key,
                    api_url=args.api_url,
                    image_path=img_path,
                    coco_label=category,
                    modification_goal=goal,
                    temperature=args.temperature
                )
                if "[Error]" in instr["object_caption"]:
                    print(f"Error on {img_path} ({goal}): {instr['object_caption']}")
                    continue

                record = {
                    "category": category,
                    "image_path": img_path,
                    "modification_goal": goal,
                    **instr
                }
                results.append(record)
                processed.add(key)

                # Incremental save
                with open(args.output_json, "w", encoding="utf-8") as out:
                    json.dump(results, out, indent=2, ensure_ascii=False)

    print("First pass completed.")


def second_pass(args):
    """
    Second pass: de-duplicate added objects for addition/replacement goals.
    Retries up to 3 times with exclusion; drops if still duplicate.
    """
    if not os.path.isfile(args.output_json):
        print(f"File not found: {args.output_json}")
        return

    with open(args.output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    used = set()
    final = []

    for item in tqdm(data, desc="Deduplication"):
        if "[Error]" in item.get("object_caption", ""):
            continue

        goal = item["modification_goal"]
        if goal not in ("object addition", "object replacement"):
            final.append(item)
            continue

        sym = item.get("symbolic_modification", "")
        obj = parse_added_object(sym) or ""
        if not obj or obj not in used:
            used.add(obj)
            final.append(item)
        else:
            # Retry up to 3 times
            for _ in range(3):
                new = generate_modification_instructions(
                    api_key=args.api_key,
                    api_url=args.api_url,
                    image_path=item["image_path"],
                    coco_label=item["category"],
                    modification_goal=goal,
                    temperature=args.temperature,
                    exclude_object=obj
                )
                new_obj = parse_added_object(new.get("symbolic_modification", "")) or ""
                if new_obj and new_obj not in used:
                    used.add(new_obj)
                    # Apply updates
                    for k in new:
                        item[k] = new[k]
                    final.append(item)
                    break
            # if still duplicate after retries, drop it

    # Save final results
    with open(args.output_json, "w", encoding="utf-8") as out:
        json.dump(final, out, indent=2, ensure_ascii=False)

    print("Second pass completed.")


# ------------------------------
# CLI Entry Point
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-generate COCO image modification instructions"
    )
    parser.add_argument(
        "--api-key", required=True,
        help="Your GPT-4o API key"
    )
    parser.add_argument(
        "--api-url",
        default="https://gpt.hkust-gz.edu.cn/v1/chat/completions",
        help="GPT API endpoint URL"
    )
    parser.add_argument(
        "--base-dir", required=True,
        help="Root directory of COCO images (subfolders per category)"
    )
    parser.add_argument(
        "--output-json", required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--modification-goals", nargs="+",
        default=["object addition", "object replacement"],
        help="List of modification goals to generate, separated by spaces"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (0.0–2.0)"
    )
    parser.add_argument(
        "--skip-second-pass", action="store_true",
        help="Only run the first pass; skip deduplication"
    )
    args = parser.parse_args()

    first_pass(args)
    if not args.skip_second_pass:
        second_pass(args)
