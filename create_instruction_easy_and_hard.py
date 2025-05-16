import os
import re
import json
import base64
import requests
from tqdm import tqdm

# ------------------------------
# Configuration
# ------------------------------
API_KEY = "###"
API_URL = "https://gpt.hkust-gz.edu.cn/v1/chat/completions"
BASE_DIR = "/fdata/FragFake/coco_train_sample_3_25"    # adjust if needed
OUTPUT_JSON = "/fdata/FragFake/finished_file/easy_instructions.json"
MODIFICATION_GOALS = ["object addition", "object replacement"]


# ------------------------------
# Helper Functions
# ------------------------------
def encode_image(image_path):
    """Read image and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_gpt_response(text):
    """Strip Markdown code fences from GPT response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop opening and closing fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_added_object(symbolic_modification):
    """
    From a string like '-refrigerator +dishwasher', return 'dishwasher'.
    Returns None if no '+...' is found.
    """
    match = re.search(r"\+(\S.*)$", symbolic_modification)
    return match.group(1).strip() if match else None


def generate_modification_instructions(
    image_path, coco_label, modification_goal, exclude_object=None
):
    """
    Call GPT-4o Vision to generate:
      - object_caption
      - brief_modification_instruction
      - descriptive_modification
      - symbolic_modification
    If exclude_object is given, add a constraint.
    """
    # build exclusion clause
    exclude_clause = (
        f"\n\nImportant: Please do NOT use the following object: {exclude_object}\n"
        if exclude_object else ""
    )

    prompt = f"""### Task: Advanced Image Modification Instruction Generation

**Description:**
Given a COCO object label in the image, generate four fields:
1. **object_caption**: A natural description of the object.
2. **brief_modification_instruction**: A concise, imperative edit (e.g., "Add a fighter jet.").
3. **descriptive_modification**: The merged caption + new edit.
4. **symbolic_modification**: A short '+obj' / '-obj' string.

{exclude_clause}
---
**Input:**
- **COCO Object Label:** {coco_label}
- **Image:** [provided inline]
- **Modification Goal:** {modification_goal}

---
**Output (valid JSON):**
{{
  "object_caption": "",
  "brief_modification_instruction": "",
  "descriptive_modification": "",
  "symbolic_modification": ""
}}
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    image_b64 = encode_image(image_path)

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    }
                ]
            }
        ],
        "temperature": 1.0,
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        cleaned = clean_gpt_response(content)
        parsed = json.loads(cleaned)

        # validate keys
        expected = {
            "object_caption",
            "brief_modification_instruction",
            "descriptive_modification",
            "symbolic_modification",
        }
        if expected.issubset(parsed):
            return {k: parsed[k] for k in expected}
        else:
            return {
                "object_caption": "[Error] Missing keys in JSON.",
                "brief_modification_instruction": cleaned,
                "descriptive_modification": "",
                "symbolic_modification": ""
            }

    except requests.HTTPError as e:
        return {
            "object_caption": f"[Error] API request failed: {e}",
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
# Main Processing Functions
# ------------------------------
def main():
    """First pass: generate instructions for each image + goal."""
    processed = set()
    results = []

    # load existing results (for retrying errors)
    if os.path.isfile(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for item in existing:
                if "[Error]" not in item.get("object_caption", ""):
                    key = (item["image_path"], item["modification_goal"])
                    processed.add(key)
                    results.append(item)
        except json.JSONDecodeError:
            print(f"[Warning] Could not parse {OUTPUT_JSON}, starting fresh.")

    # walk through images
    for category in tqdm(os.listdir(BASE_DIR), desc="Categories"):
        cat_path = os.path.join(BASE_DIR, category)
        if not os.path.isdir(cat_path):
            continue

        for img in tqdm(os.listdir(cat_path), desc=f"Images in {category}", leave=False):
            img_path = os.path.join(cat_path, img)
            if not os.path.isfile(img_path):
                continue

            for goal in MODIFICATION_GOALS:
                key = (img_path, goal)
                if key in processed:
                    continue

                instr = generate_modification_instructions(img_path, category, goal)
                if "[Error]" in instr["object_caption"]:
                    print(f"Error: {instr['object_caption']} on {img_path} ({goal})")
                    continue

                record = {
                    "category": category,
                    "image_path": img_path,
                    "modification_goal": goal,
                    **instr
                }
                results.append(record)
                processed.add(key)

                # incremental save
                with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
                    json.dump(results, out, indent=2, ensure_ascii=False)

    print("Initial generation completed.")


def second_pass():
    """
    Second pass: de-duplicate added objects for addition/replacement goals.
    Retries up to 3 times with exclusion; drops if still duplicate.
    """
    if not os.path.isfile(OUTPUT_JSON):
        print(f"File not found: {OUTPUT_JSON}")
        return

    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    used = set()
    final = []

    for idx, item in enumerate(tqdm(data, desc="Deduplication"), start=1):
        if "[Error]" in item.get("object_caption", ""):
            continue

        goal = item["modification_goal"]
        if goal not in ("object addition", "object replacement"):
            final.append(item)
        else:
            sym = item.get("symbolic_modification", "")
            obj = parse_added_object(sym) or ""
            if not obj or obj not in used:
                used.add(obj)
                final.append(item)
            else:
                # retry up to 3 times
                for attempt in range(1, 4):
                    new = generate_modification_instructions(
                        item["image_path"], item["category"], goal, exclude_object=obj
                    )
                    new_obj = parse_added_object(new.get("symbolic_modification", "")) or ""
                    if new_obj and new_obj not in used:
                        used.add(new_obj)
                        # update item
                        for k in new:
                            item[k] = new[k]
                        final.append(item)
                        break
                # if still duplicate after retries, drop silently

        # incremental save
        with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
            json.dump(final, out, indent=2, ensure_ascii=False)

    print("Second pass completed.")


if __name__ == "__main__":
    main()
    second_pass() # remove this function is the easy version
