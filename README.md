# LoCCA-Frame

Official implementation of **LoCCA-Frame**, a mental health counseling framework presented at **KCC 2025**.

This project combines **MentalRoBERTa-based disorder classification** with **GPT-4o response generation** to provide strategy-driven counseling.

## Paper

**Title**: LoCCA-Frame: Low-Cost Strategic Prompt Framework for Improving Mental Health Counseling Competence Based on Disorder Prediction  
(LoCCA-Frame: 질환 예측 기반 정신 건강 상담 역량 향상을 위한 저비용 전략 프롬프트 프레임워크)

**Authors**: Kwangil Kim, Seulgi Kim, Harksoo Kim  
**Conference**: Korea Computer Congress 2025 (KCC 2025)

### Abstract

Generic LLMs often hallucinate or provide ungrounded advice in mental health contexts. This framework addresses these issues using a two-stage pipeline:

1.  **Classification**: A fine-tuned MentalRoBERTa model identifies the user's specific mental disorder (e.g., Depression, Anxiety).
2.  **Generation**: GPT-4o generates a response conditioned on clinically verified treatment strategies associated with the detected disorder.

## Project Structure

```
LoCCA-Frame/
├── src/
│   ├── MentalRoBERTa_Training.py   # Fine-tuning script for classification
│   ├── pipeline.py                 # Main inference pipeline
│   └── Trained_MentalRoBERTa/      # Saved model checkpoints
├── data/
│   ├── strategy_info.json          # Mapping of disorders to treatment strategies
│   └── inference_sample.jsonl      # Sample input data
└── README.md
```

## Requirements

  * Python 3.8+
  * PyTorch
  * Transformers
  * OpenAI API

Install dependencies:

```bash
pip install torch transformers scikit-learn scikit-multilearn pandas numpy tqdm openai attrdict
```

## Usage

### 1\. Classification Model Training

Fine-tune MentalRoBERTa on the mental health dataset.

```bash
python src/MentalRoBERTa_Training.py
```

The model will be saved to `./src/Trained_MentalRoBERTa/best_model/`.

### 2\. Inference Pipeline

Run the full pipeline (Classification + Strategy Retrieval + Generation).

```bash
python src/pipeline.py --api_key YOUR_OPENAI_API_KEY --data_file ./data/your_data.jsonl
```

**Arguments:**

  * `--api_key`: (Required) OpenAI API key.
  * `--data_file`: Path to input JSONL file.
  * `--output_file`: Name of the output file (default: `total_inference.jsonl`).

## Data

The model was trained on a combined dataset of **SWMH**, **Reddit Mental Health Diagnoses**, and **Comprehensive PTSD Analysis Dataset**.

**Classes**: Depression, Anxiety, PTSD, Bipolar Disorder, Eating Disorder.

> Note: Raw datasets are not included in this repository due to licensing/privacy restrictions.

## Citation

```bibtex
@inproceedings{Kim2025LoCCA,
  title={LoCCA-Frame: 질환 예측 기반 정신 건강 상담 역량 향상을 위한 저비용 전략 프롬프트 프레임워크 (LoCCA-Frame: Low-Cost Strategic Prompt Framework for Improving Mental Health Counseling Competence Based on Disorder Prediction)},
  author={Kim, Sumin and Kim, Seulgi and Kim, Harksoo},
  booktitle={Proceedings of the Korea Computer Congress 2025 (2025 한국컴퓨터종합학술대회 논문집)},
  pages={566--568},
  year={2025},
  organization={Korean Institute of Information Scientists and Engineers (한국정보과학회)}
}
```