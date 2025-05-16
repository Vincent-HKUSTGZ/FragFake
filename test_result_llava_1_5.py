import os
import json
import re
import torch
import argparse
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# ============ 规则判断函数 ============
def extract_groundtruth_object(text):
    match = re.search(r"The (.*?) in the image has been modified\.", text, re.IGNORECASE)
    return match.group(1).lower().strip() if match else ""

def extract_predicted_object(text):
    patterns = [
        r"The (.*?)(?: has been| was| is)(?: modified| added)",
        r"Added (?:a |an |the )?(.*?)\.",
        r"Modified (?:the )?(.*?)\.",
        r"the (.*?) (?:is|was) (?:modified|added)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower().strip()
    return ""

# ============ 推理函数 ============
def inference_llava(sample, model, processor):
    image = Image.open(sample['images'][0])
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample['messages'][0]['content'].replace('<image>', '')},
                {"type": "image"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
    input_ids_len = inputs.input_ids.shape[1]
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids = output[0][input_ids_len:]
    return processor.decode(generated_ids, skip_special_tokens=True)

# ============ 辅助函数 ============
def find_original_image_path(original_filename, base_dir="/fdata/FragFake/coco_train_sample_3_25"):
    for root, dirs, files in os.walk(base_dir):
        if original_filename in files:
            return os.path.join(root, original_filename)
    return None

# ============ 主函数 ============
def evaluate(model_id, input_path, output_path):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    with open(input_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    evaluation_results = []

    for idx, sample in enumerate(test_data):
        print(f"Processing sample {idx+1}")

        model_output = inference_llava(sample, model, processor).lower()
        ground_truth_text = sample["messages"][1]["content"].lower()

        gt_label = "original" if "nothing has been modified" in ground_truth_text else "modified"
        pred_label = "original" if "nothing has been modified" in model_output else "modified"

        sample_result = {
            "sample_index": idx,
            "ground_truth_label": gt_label,
            "predicted_label": pred_label,
            "model_output": model_output,
            "ground_truth": ground_truth_text,
            "original_image_path": None,
            "modified_image_path": None,
            "is_modified_correct": "yes" if gt_label == pred_label else "no",
            "is_object_correct": "no"
        }

        if gt_label == "original":
            sample_result["original_image_path"] = sample["images"][0]
        else:
            sample_result["modified_image_path"] = sample["images"][0]
            filename = os.path.basename(sample["images"][0])
            original_filename = filename.split('_')[0] + ".jpg"
            sample_result["original_image_path"] = find_original_image_path(original_filename)

        if gt_label == "modified" and pred_label == "modified":
            gt_obj = extract_groundtruth_object(ground_truth_text)
            pred_obj = extract_predicted_object(model_output)
            sample_result["is_object_correct"] = "yes" if gt_obj and (gt_obj in pred_obj or pred_obj in gt_obj) else "no"

        evaluation_results.append(sample_result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation completed. Results saved to {output_path}")

# ============ 入口 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLaVA evaluation on a dataset.")
    parser.add_argument("--model_id", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the evaluation results.")
    args = parser.parse_args()

    evaluate(args.model_id, args.input_path, args.output_path)
