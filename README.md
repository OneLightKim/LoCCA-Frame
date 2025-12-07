제공해주신 정보를 바탕으로 **LoCCA-Frame**의 연구 배경과 논문의 핵심 내용을 포함하여 전문적인 **README.md** 초안을 작성했습니다.

기존 내용에 **"📝 About the Paper"** 섹션을 추가하여 이 프레임워크가 왜 제안되었는지(LLM의 환각 현상 완화, 전문적인 상담 전략 통합 등)를 학술적으로 설명했습니다. 또한, 파일명 오타(`Metal` -\> `Mental`)를 수정하고 가독성을 높였습니다.

아래 내용을 복사하여 사용하시면 됩니다.

-----

````markdown
# LoCCA-Frame: Strategy-Driven Mental Health Counseling Framework

**LoCCA-Frame** is a two-stage mental health counseling framework that integrates **MentalRoBERTa-based disorder classification** with **GPT-4o powered response generation**. This repository contains the official implementation of the research aimed at providing empathetic and clinically grounded counseling responses.

## 📝 About the Paper

**Title:** [Insert Your Paper Title Here]  
**Authors:** [Insert Author Names]  
**Publication:** [Insert Conference/Journal Name, e.g., Expert Systems With Applications]

### Abstract & Motivation
Recent advancements in Large Language Models (LLMs) have shown promise in mental health support. However, generic LLMs often struggle with specific clinical contexts and may generate hallucinatory or ungrounded advice. 

**LoCCA-Frame** addresses these limitations by introducing a pipeline approach:
1.  **Disorder Identification**: Instead of relying solely on the LLM's implicit knowledge, we utilize a specialized encoder (MentalRoBERTa) to explicitly classify the user's mental state into specific categories (e.g., Depression, Anxiety, PTSD).
2.  **Strategy Integration**: Based on the classification, the system retrieves clinically verified treatment strategies defined in our dataset.
3.  **Grounded Generation**: The LLM (GPT-4o) generates a response that is not only empathetic but strictly adheres to the retrieved counseling strategies, ensuring higher factual consistency and safety.

## 🏗️ Framework Architecture

The system operates in a sequential pipeline:

1.  **Classification Stage (MentalRoBERTa)**
    * **Input**: User query/post.
    * **Model**: Fine-tuned `mental/mental-roberta-base`.
    * **Output**: Predicted mental disorder class (Depression, Anxiety, PTSD, Bipolar Disorder, Eating Disorder).

2.  **Response Generation Stage (GPT-4o)**
    * **Context Injection**: The predicted class is mapped to a specific counseling strategy (from `strategy_info.json`).
    * **Generation**: GPT-4o synthesizes the user's query with the retrieved strategy to produce the final counseling response.

## 📂 Project Structure

```bash
LoCCA-Frame/
├── src/
│   ├── MentalRoBERTa_Training.py   # Script for fine-tuning the classification model
│   ├── pipeline.py                 # Inference pipeline (Classification -> Strategy Retrieval -> Generation)
│   └── Trained_MentalRoBERTa/      # Directory for saving trained model checkpoints
├── data/
│   ├── strategy_info.json          # Dictionary mapping disorders to treatment strategies
│   └── [dataset].jsonl             # Input data for training/inference
└── README.md
````

## 🛠️ Requirements

Ensure you have the following dependencies installed:

```txt
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

You can install them via pip:

```bash
pip install torch transformers scikit-learn pandas numpy tqdm openai attrdict
```

## 🚀 Usage

### 1\. Train the Classification Model

Fine-tune the MentalRoBERTa model on the combined mental health dataset.

```bash
python src/MentalRoBERTa_Training.py
```

> **Output**: The best-performing model will be saved to `./src/Trained_MentalRoBERTa/best_model/`.

### 2\. Run Inference Pipeline

Generate counseling responses using the trained classifier and OpenAI API.

```bash
python src/pipeline.py --api_key YOUR_OPENAI_API_KEY --data_file ./data/your_data.jsonl
```

#### Arguments

| Argument | Default | Description |
|:---|:---|:---|
| `--api_key` | **Required** | Your OpenAI API Key |
| `--data_file` | `./data/v2_finaldf123_inference_sample.jsonl` | Path to the input JSONL file containing user queries |
| `--output_dir` | `./data` | Directory to save results |
| `--output_file` | `total_inference.jsonl` | Filename for the generated responses |

## 📊 Datasets

> **Note**: Due to privacy regulations and platform policies, the raw datasets used in this research are not included in this repository. Please refer to the original sources below.

### Training Data (Classification)

We constructed a balanced dataset by integrating three sources:

  * **SWMH** (Stop Words Mental Health)
  * **Reddit Mental Health Diagnoses**
  * **Comprehensive PTSD Analysis Dataset**

**Class Distribution:**
| Disorder | Count |
|:---|---:|
| Depression | 11,940 |
| Anxiety | 6,136 |
| PTSD | 5,084 |
| Bipolar Disorder | 4,932 |
| Eating Disorder | 426 |

### Inference Data (Generation)

  * **Source**: CounselChat (verified questions asked to counselors).
  * **Scale**: 609 query texts.
  * **Evaluation**: Quantitative evaluation via GPT-based metrics and qualitative evaluation by human experts.

## 📜 License

This project is intended for **research purposes only**. The system is not a substitute for professional medical advice, diagnosis, or treatment.

## 🖊️ Citation

If you find this code or our paper useful for your research, please cite:

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

*(Note: Please update the BibTeX entry with the exact details of your publication once available.)*