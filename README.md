# FragFake: A Dataset for Fine-Grained Detection of Edited Images with Vision Language Models

This a official repository for FragFake: A Dataset for Fine-Grained Detection of Edited Images with Vision Language Models

## Repository Structure

This repository contains several key Python scripts for working with the FragFake dataset:

- **clean_instruction_create_format.py**: Processes the raw data to generate conversation format for training and testing
- **test_gemma.py**: Evaluation script for Gemma models
- **test_result_llava_1_5.py**: Evaluation script for LLaVA models
- **test_result_qwen2.py**: Evaluation script for Qwen2 models
- **test_result_qwen2_5.py**: Evaluation script for Qwen2.5 models
- **create_instruction_easy_and_hard.py**: Instructions creation
## Usage

### Data Processing

The `clean_instruction_create_format.py` script processes raw data from the FragFake dataset to create conversation pairs for both training and testing:

```bash
python clean_instruction_create_format.py
```

This script:
1. Extracts objects from symbolic modifications
2. Removes duplicates in the conversations
3. Generates conversation pairs for original and modified images
4. Splits the data into training and testing sets

### Model Evaluation

We provide evaluation scripts for multiple VLM architectures. Each script loads a fine-tuned model, runs inference on the test dataset, and saves the evaluation results.

#### Evaluating Gemma Models

```bash
python test_gemma.py
```

#### Evaluating LLaVA Models

```bash
python test_result_llava_1_5.py --model_id "/path/to/model" --input_path "/path/to/test_data.json" --output_path "/path/to/save/results.json"
```

#### Evaluating Qwen2 Models

```bash
python test_result_qwen2.py
```

#### Evaluating Qwen2.5 Models

```bash
python test_result_qwen2_5.py
```

#### Creating Instruction
```bash
python generate_instructions.py \
  --api-key YOUR_GPT4O_API_KEY \
  --base-dir /fdata/FragFake/coco_train_sample_3_25 \
  --output-json /fdata/FragFake/finished_file/easy_instructions.json \
  --modification-goals "object addition" "object replacement" \
  --temperature 1.0

--api-key (required): your GPT-4o API key

--api-url: (optional) the API endpoint; defaults to the HKUST-GZ URL

--base-dir (required): path to COCO image root directory (organized by category)

--output-json (required): path where the result JSON will be saved

--modification-goals: (optional) space-separated list of goals; defaults to object addition object replacement

--temperature: (optional) sampling temperature (0.0–2.0), default 1.0

--skip-second-pass: (optional) if set, only the first pass runs, which is Easy version (no deduplication)
```

## Model Fine-tuning

All model fine-tuning in this repository is performed using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), a comprehensive framework for fine-tuning large language models. Please refer to the LLaMA-Factory documentation for detailed instructions on setting up and running the fine-tuning process.
