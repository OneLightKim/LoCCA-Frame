# LoCCA-Frame

A mental health counseling framework that combines **MentalRoBERTa-based disorder classification** with **GPT-powered response generation**.

## Overview

This project implements a two-stage pipeline for mental health counseling:
1. **Classification Stage**: Fine-tuned MentalRoBERTa model classifies user posts into mental disorder categories
2. **Response Generation Stage**: GPT-4o generates empathetic counseling responses based on the predicted disorder and corresponding treatment strategies

## Project Structure

```
LoCCA-Frame/
├── src/
│   ├── MetalRoBERTa_Taining.py   # MentalRoBERTa fine-tuning script
│   └── pipeline.py                # Inference pipeline (classification + generation)
├── data/
│   └── strategy_info.json         # Treatment strategies for each disorder
└── README.md
```

## Requirements

```
torch
transformers
scikit-learn
scikit-multilearn
pandas
numpy
tqdm
openai
attrdict
```

## Usage

### 1. Train the Classification Model

```bash
python src/MetalRoBERTa_Taining.py
```

The trained model will be saved to `./src/Trained_MentalRoBERTa/best_model/`.

### 2. Run Inference Pipeline

```bash
python src/pipeline.py --api_key YOUR_OPENAI_API_KEY --data_file ./data/your_data.jsonl
```

#### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--api_key` | (required) | OpenAI API key |
| `--data_file` | `./data/v2_finaldf123_inference_sample.jsonl` | Input data file path |
| `--output_dir` | `./data` | Output directory |
| `--output_file` | `total_inference.jsonl` | Output file name |

## Dataset

> **Note**: The datasets used in this research are not included in this repository due to access restrictions. Please contact the original dataset authors for access.

### Training Data (Classification Model)
We integrated three Reddit and social media-based datasets for building the mental disorder classification model:
- **SWMH** (Stop Words Mental Health)
- **Reddit Mental Health Diagnoses**
- **Comprehensive PTSD Analysis Dataset**

#### Class Distribution
| Disorder | Samples |
|----------|---------|
| Depression | 11,940 |
| Anxiety | 6,136 |
| PTSD | 5,084 |
| Bipolar Disorder | 4,932 |
| Eating Disorder | 426 |

*Note: Class imbalance exists in the dataset. The data was split in a 4:1 ratio for training and evaluation.*

### Inference Data (Response Generation)
- **CounselChat**: A dataset of social media users asking questions to counselors
- 609 query texts in total
- GPT-based automatic evaluation on full dataset
- Qualitative evaluation on 15 randomly sampled examples

## Model

- **Base Model**: [mental/mental-roberta-base](https://huggingface.co/mental/mental-roberta-base)
- **Classification**: 5 classes (Depression, Anxiety, Bipolar, Eating Disorder, and one additional class)
- **Generation**: GPT-4o with disorder-specific treatment strategies

## License

This project is for research purposes only.

## Citation

If you use this code, please cite our work.

