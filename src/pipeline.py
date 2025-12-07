import os
import json
import argparse
from tqdm import tqdm
from attrdict import AttrDict
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_utils import PreTrainedModel
import torch.nn as nn
from openai import OpenAI

# --------------------------------------------------------------------------------
# 1. ChatGPT API 호출을 위한 함수 (첫 번째 스크립트에서 가져옴)
# --------------------------------------------------------------------------------
def create_chat_completion(client, system_prompt, user_prompt, model="gpt-4o", temperature=1.2, max_tokens=400):
    """
    OpenAI의 Chat Completion API를 호출하는 함수.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # 응답이 유효한지 확인 후 content 반환
        if response and response.choices:
            return response.choices[0].message.content.strip()
        else:
            print("Warning: Invalid response structure from API.")
            return "Error: Invalid response from API."
    except Exception as e:
        print(f"Error during API call: {e}")
        return f"Error: {str(e)}"

# --------------------------------------------------------------------------------
# 2. RoBERTa 분류 모델 관련 함수 (기존과 동일)
# --------------------------------------------------------------------------------
class MentalROBERTA_SENTIMENT_CLASSIFIER(PreTrainedModel):
    def __init__(self, config):
        super(MentalROBERTA_SENTIMENT_CLASSIFIER, self).__init__(config)
        self.roberta = AutoModel.from_pretrained(config._name_or_path)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_vector)
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        logits = self.linear2(x)
        return logits

def load_roberta_model(model_dir):
    try:
        roberta_config = AutoConfig.from_pretrained(model_dir)
        setattr(roberta_config, "num_labels", 5)
        setattr(roberta_config, "hidden_size", 768)
        model = MentalROBERTA_SENTIMENT_CLASSIFIER.from_pretrained(
            model_dir, config=roberta_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        raise Exception(f"RoBERTa 모델 로딩 중 오류 발생: {str(e)}")

def predict_label(model, tokenizer, device, sentence):
    try:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probs = softmax(outputs, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
        return pred
    except Exception as e:
        raise Exception(f"레이블 예측 중 오류 발생: {str(e)}")

# --------------------------------------------------------------------------------
# 3. 메인 파이프라인 함수 (LLaMA 호출 부분을 ChatGPT 호출로 변경)
# --------------------------------------------------------------------------------
def main(cli_args):
    try:
        args = AttrDict(vars(cli_args))
        
        # --- LLaMA 모델 로딩 제거 ---
        # --- OpenAI 클라이언트 초기화 추가 ---
        client = OpenAI(api_key=args.api_key)
        
        # RoBERTa 모델 로딩
        roberta_model, roberta_tokenizer, device = load_roberta_model("./src/Trained_MentalRoBERTa/best_model")

        # 레이블 매핑 로드
        with open("./src/Trained_MentalRoBERTa/best_model/label2idx.json", 'r', encoding='utf-8') as f:
            label2idx = json.load(f)
        idx2label = {str(v): k for k, v in label2idx.items()}

        # 데이터셋 로드
        with open(args.data_file, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]

        # 전략 정보 로드
        with open("./data/strategy_info.json", 'r', encoding='utf-8') as f:
            strategy_info = json.load(f)

        # 출력 디렉토리 생성
        os.makedirs(args.output_dir, exist_ok=True)
        # 출력 파일을 하나로 통일하여 간결하게 만듦
        output_path = os.path.join(args.output_dir, args.output_file)

        with open(output_path, 'w', encoding='utf-8') as of:
            for user_data in tqdm(dataset, desc="Inference with ChatGPT"):
                try:
                    required_fields = ["questionID", "questionTitle", "questionText"]
                    if not all(field in user_data for field in required_fields):
                        print(f"Warning: Missing required fields in data: {user_data.get('questionID', 'unknown')}")
                        continue

                    user = user_data["questionID"]
                    post = f"{user_data['questionTitle']}. {user_data['questionText']}"
                    answer = user_data.get("answerText", "")

                    # 1. RoBERTa로 레이블 예측 (기존과 동일)
                    pred_label_idx = predict_label(roberta_model, roberta_tokenizer, device, post)
                    pred_label = idx2label.get(str(pred_label_idx), "unknown")
                    strategy = strategy_info.get(pred_label, "No strategy available.")

                    # 2. 프롬프트 생성 (시스템 프롬프트와 사용자 프롬프트 분리)
                    system_prompt = (
                        "You are a mental health counseling expert. You will analyze each user's counseling content, "
                        "and provide responses aimed at treating their mental disorder condition based on the user's "
                        "mental disorder information and appropriate strategies for that condition."
                    )
                    user_prompt = (
                        f"The user's counseling content is as follows: {post}\n\n"
                        f"Based on my analysis, the user's mental disorder is '{pred_label}', "
                        f"and the corresponding treatment strategy for this condition is: {strategy}\n\n"
                        "Please provide a response aimed at treating the user's mental disorder by utilizing the user's counseling content, mental disorder information, and the corresponding treatment strategy. "
                        "The response should be empathetic, supportive, and practical."
                    )

                    # 3. ChatGPT API 호출로 텍스트 생성
                    generated = create_chat_completion(client, system_prompt, user_prompt, model="gpt-4o", max_tokens=400)

                    # 4. 결과 저장
                    result = {
                        "questionID": user,
                        "total_questionText": post,
                        "model_answer": generated,
                        "answerText": answer,
                        "predicted_label": pred_label, # 'goldenLabel'에서 이름 변경
                        "applied_strategy": strategy # 'strategy'에서 이름 변경
                    }
                    of.write(json.dumps(result, ensure_ascii=False) + "\n")

                    # GPU 메모리 정리 (RoBERTa 사용 후)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error processing data for questionID {user_data.get('questionID', 'unknown')}: {str(e)}")
                    continue

        print(f"Results successfully saved to {output_path}")

    except Exception as e:
        print(f"An error occurred in the main execution: {str(e)}")
        raise

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Run inference using RoBERTa for classification and ChatGPT for generation.")
    
    # 필수 인자
    cli_parser.add_argument("--api_key", type=str, default="", help="Your OpenAI API key.")
    # 파일 경로 인자
    cli_parser.add_argument("--data_file", type=str, default="./data/v2_finaldf123_inference_sample.jsonl", help="Path to the input data file.")
    cli_parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save the output file.")
    cli_parser.add_argument("--output_file", type=str, default="total_inference.jsonl", help="Name of the output file.")
    
    args = cli_parser.parse_args()
    
    
    # API 키를 환경 변수로 설정하는 것이 더 안전하지만, 인자로 직접 전달
    # os.environ["OPENAI_API_KEY"] = args.api_key
    
    main(args)