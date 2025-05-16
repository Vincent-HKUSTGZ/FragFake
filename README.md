# FragFake

A dataset and framework for fine-grained detection of edited images using Vision Language Models (VLMs).

## Overview

FragFake is the first dedicated benchmark dataset for edited image detection, focused on addressing key challenges in image manipulation detection:

1. Moving beyond binary classification to precise localization of edits
2. Avoiding costly pixel-level annotations required by traditional computer vision methods
3. Providing a large-scale, high-quality dataset for modern image-editing detection techniques

## Repository Structure

This repository contains several key Python scripts for working with the FragFake dataset:

- **clean_instruction_create_format.py**: Processes the raw data to generate conversation format for training and testing
- **test_gemma.py**: Evaluation script for Gemma models
- **test_result_llava_1_5.py**: Evaluation script for LLaVA models
- **test_result_qwen2.py**: Evaluation script for Qwen2 models
- **test_result_qwen2_5.py**: Evaluation script for Qwen2.5 models

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

## Model Fine-tuning

All model fine-tuning in this repository is performed using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), a comprehensive framework for fine-tuning large language models. Please refer to the LLaMA-Factory documentation for detailed instructions on setting up and running the fine-tuning process.

### Fine-tuning Process

1. Prepare the training dataset in the conversation format using `clean_instruction_create_format.py`
2. Use LLaMA-Factory to fine-tune the model of your choice
3. Evaluate the fine-tuned model using the appropriate test script

## Evaluation Metrics

The evaluation scripts measure:
1. **Classification Accuracy**: Ability to identify whether an image has been modified
2. **Object Localization**: Precision in identifying which objects were modified
