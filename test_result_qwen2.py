import os
import json
import re
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ============ 新增：规则判断函数 ============
def extract_groundtruth_object(text):
    """从ground truth中提取物体"""
    match = re.search(r"The (.*?) in the image has been modified\.", text, re.IGNORECASE)
    return match.group(1).lower().strip() if match else ""

def extract_predicted_object(text):
    """从模型输出中提取物体"""
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

# ============ Qwen2.5 推理部分保持不变 ============
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/fdata/FragFake/LLaMA-Factory-pre2/output/gemini_easy_qwen2_original_coco", 
    torch_dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("/data_sda/models/Qwen2-VL-7B-Instruct")
input_path = "/fdata/FragFake/finished_file/dataset/gemini/easy/gemini_easy_test_conversation.json"
output_path = "/fdata/FragFake/finished_file/dataset/gemini/result/gemini_easy_qwen2_original_vl_coco.json"
def inference_qwen(sample):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{sample['images'][0]}",
                },
                {"type": "text", "text": f"{sample['messages'][0]['content'].replace('<image>','')}"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# ============ 辅助函数保持不变 ============
def find_original_image_path(original_filename, base_dir="/fdata/FragFake/coco_train_sample_3_25"):
    for root, dirs, files in os.walk(base_dir):
        if original_filename in files:
            return os.path.join(root, original_filename)
    return None

# ============ 测试集处理逻辑修改 ============
with open(input_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

evaluation_results = []

for idx, sample in enumerate(test_data):
    print(f"Processing sample {idx+1}")
    
    # 模型推理
    model_output = inference_qwen(sample).lower()
    ground_truth_text = sample["messages"][1]["content"].lower()
    
    # 标签判断
    gt_label = "original" if "nothing has been modified" in ground_truth_text else "modified"
    pred_label = "original" if "nothing has been modified" in model_output else "modified"
    
    # 初始化结果
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
    
    # 路径处理
    if gt_label == "original":
        sample_result["original_image_path"] = sample["images"][0]
    else:
        sample_result["modified_image_path"] = sample["images"][0]
        filename = os.path.basename(sample["images"][0])
        original_filename = filename.split('_')[0] + ".jpg"
        sample_result["original_image_path"] = find_original_image_path(original_filename)
    
    # 物体判断逻辑
    if gt_label == "modified" and pred_label == "modified":
        gt_obj = extract_groundtruth_object(ground_truth_text)
        pred_obj = extract_predicted_object(model_output)
        sample_result["is_object_correct"] = "yes" if gt_obj and (gt_obj in pred_obj or pred_obj in gt_obj) else "no"
    
    evaluation_results.append(sample_result)

# 保存结果
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

print(f"Evaluation completed. Results saved to {output_path}")