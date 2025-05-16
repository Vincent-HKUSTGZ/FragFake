import os
import json
import re
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# ============ Gemma‑3‑4B‑IT 加载 ============
model_id = "/fdata/FragFake/LLaMA-Factory-pre2/output/gemma_4b_gemini_easy_train_3000_2"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto"
).eval()                                       # 推理模式
processor = AutoProcessor.from_pretrained(model_id)

# ============ ground‑truth / 预测物体抽取函数（保持不变） ============
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

# ============ Gemma 推理函数 ============
def inference_gemma(sample):
    # 构造符合 Gemma 多模态聊天格式的 messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["images"][0]},
                {"type": "text",  "text": sample["messages"][0]["content"].replace("<image>", "")}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device, dtype=torch.bfloat16)   # Gemma 推荐 bfloat16

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )
    generated_ids = generated_ids[:, input_len:]        # 去掉提示部分
    return processor.decode(generated_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)

# ============ 其它辅助函数保持不变 ============
def find_original_image_path(original_filename, base_dir="/fdata/FragFake/coco_train_sample_3_25"):
    for root, dirs, files in os.walk(base_dir):
        if original_filename in files:
            return os.path.join(root, original_filename)
    return None

# ------------- 数据路径 -------------
input_path = "/fdata/FragFake/finished_file/dataset/gemini/easy/gemini_easy_test_conversation.json"
output_path = "/fdata/FragFake/finished_file/dataset/gemini/result/gemini_easy_gemma3_4b_4000.json"

# ============ 测试集循环与评估（仅把 inference_qwen → inference_gemma） ============
with open(input_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

evaluation_results = []

for idx, sample in enumerate(test_data):
    print(f"Processing sample {idx+1}")

    # Gemma 推理
    model_output = inference_gemma(sample).lower()
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

    # 物体判断
    if gt_label == "modified" and pred_label == "modified":
        gt_obj = extract_groundtruth_object(ground_truth_text)
        pred_obj = extract_predicted_object(model_output)
        sample_result["is_object_correct"] = "yes" if gt_obj and (gt_obj in pred_obj or pred_obj in gt_obj) else "no"

    evaluation_results.append(sample_result)

# 保存结果
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

print(f"Evaluation completed. Results saved to {output_path}")
