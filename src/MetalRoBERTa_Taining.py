import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import os
import random
import json
os.environ["TRANSFORMERS_NO_TF"] = "1"
# JSONL 데이터 로드 함수
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# PTSD 라벨을 제외한 데이터 필터링 함수
def filter_data_exclude_ptsd(data):
    """PTSD 라벨을 가진 데이터를 제외하고 반환"""
    filtered_data = []
    excluded_count = 0
    
    for entry in data:
        # 'label' 또는 'goldenLabel' 필드에서 라벨 확인
        label = entry.get('label', entry.get('goldenLabel'))
        
        # PTSD 라벨이 포함된 데이터 제외
        if isinstance(label, list):
            # 멀티라벨인 경우 PTSD가 포함되어 있으면 제외
            if 'ptsd' not in [l.lower() for l in label]:
                filtered_data.append(entry)
            else:
                excluded_count += 1
        else:
            # 단일 라벨인 경우 PTSD가 아니면 포함
            if label and label.lower() != 'ptsd':
                filtered_data.append(entry)
            else:
                excluded_count += 1
    
    print(f"PTSD 제외: {excluded_count}개 → 남은 데이터: {len(filtered_data)}개")
    
    return filtered_data

# 데이터 전처리 함수 (RoBERTa에 맞게 수정)
def convert_data2feature(datas, max_length, tokenizer, label2idx=None):
    input_ids_features, attention_mask_features, label_id_features = [], [], []

    for row in tqdm(datas, desc="convert_data2feature", total=len(datas)):
        # 'text' 필드가 없는 경우를 대비한 예외 처리
        input_sequence = row.get('text', row.get('questionText', ''))
        encoding = tokenizer.encode_plus(
            input_sequence,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids_features.append(encoding["input_ids"].squeeze(0).tolist())
        attention_mask_features.append(encoding["attention_mask"].squeeze(0).tolist())

        if label2idx is not None:
            # 'label' 필드가 없는 경우를 대비한 예외 처리
            label = row.get('label', row.get('goldenLabel'))
            if label is not None:
                label_id_features.append(label2idx[label])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids_features = torch.tensor(input_ids_features, dtype=torch.long).to(device)
    attention_mask_features = torch.tensor(attention_mask_features, dtype=torch.long).to(device)

    if label2idx is not None and label_id_features:
        label_id_features = torch.tensor(label_id_features, dtype=torch.long).to(device)
        return input_ids_features, attention_mask_features, label_id_features

    return input_ids_features, attention_mask_features

# 라벨 매핑 생성
def create_label_mapping(data):
    # 'label' 필드가 없는 경우를 대비
    labels = sorted(set(entry.get('label', entry.get('goldenLabel')) for entry in data))
    label2idx = {label: idx for idx, label in enumerate(labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label2idx, idx2label

# 시드 고정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 학습 함수
def train(config, idx2label):
    # MentalRoBERTa config 객체 생성
    roberta_config = AutoConfig.from_pretrained(config["pretrained_model_name_or_path"],
                                                   cache_dir=config["cache_dir_path"])
    setattr(roberta_config, "num_labels", config["num_labels"])
    setattr(roberta_config, "hidden_size", 768)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model_name_or_path"], cache_dir=config["cache_dir_path"])
        model = MentalROBERTA_SENTIMENT_CLASSIFIER(config=roberta_config, cache_dir=config["cache_dir_path"])
        model = model.to(device)
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        raise

    train_data = config["train_data"]
    valid_data = config["valid_data"]
    label2idx = config["label2idx"]
    
    train_input_ids, train_attention_mask, train_labels = convert_data2feature(train_data, config["max_length"], tokenizer, label2idx)
    valid_input_ids, valid_attention_mask, valid_labels = convert_data2feature(valid_data, config["max_length"], tokenizer, label2idx)

    train_features = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    train_dataloader = DataLoader(train_features, sampler=RandomSampler(train_features), batch_size=config["batch_size"])
    
    valid_features = TensorDataset(valid_input_ids, valid_attention_mask, valid_labels)
    valid_dataloader = DataLoader(valid_features, sampler=SequentialSampler(valid_features), batch_size=config["batch_size"])


    # 클래스 가중치 계산
    golden_labels = [label2idx[entry.get('label', entry.get('goldenLabel'))] for entry in train_data]
    min_weight = 0.8
    max_weight = 1.5
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(golden_labels),
        y=golden_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    max_weight = 1.5
    class_weights = torch.clamp(class_weights, min=min_weight, max=max_weight)
    
    print(f"클래스 가중치: {class_weights}")

    loss_func = nn.CrossEntropyLoss(weight=class_weights)
    
    # AdamW 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], eps=1e-5, weight_decay=0.01)

    # 학습 스케줄러 설정
    num_training_steps = len(train_dataloader) * config["epoch"]
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    max_accuracy = 0
    for epoch in range(config["epoch"]):
        model.train()
        total_loss = []

        for step, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch+1}", total=len(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            
            optimizer.zero_grad()
            hypothesis = model(input_ids, attention_mask)
            loss = loss_func(hypothesis, labels)
            
            if torch.isnan(loss):
                print("NaN loss 발생, 해당 step 건너뜀")
                optimizer.zero_grad()
                continue

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss.append(loss.item())

        avg_loss = np.mean(total_loss) if total_loss else float('nan')
        print(f"Epoch {epoch+1} 완료 - Loss: {avg_loss:.4f}")

        model.eval()
        total_hypothesis, total_labels = [], []

        for step, batch in tqdm(enumerate(valid_dataloader), desc=f"Epoch {epoch+1} Evaluation", total=len(valid_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch

            with torch.no_grad():
                hypothesis = model(input_ids, attention_mask)
                hypothesis = torch.argmax(hypothesis, dim=-1)

            hypothesis = hypothesis.cpu().detach().numpy().tolist()
            labels = labels.cpu().detach().numpy().tolist()

            total_hypothesis += hypothesis
            total_labels += labels

        accuracy = accuracy_score(total_labels, total_hypothesis)
        if max_accuracy < accuracy:
            max_accuracy = accuracy

            roberta_config.save_pretrained(save_directory=config["output_dir_path"])
            tokenizer.save_pretrained(save_directory=config["output_dir_path"])
            model.save_pretrained(save_directory=config["output_dir_path"])
            
            label_mapping = {
                "label2idx": config["label2idx"],
                "idx2label": {str(k): v for k, v in idx2label.items()}
            }
            
            with open(os.path.join(config["output_dir_path"], "label2idx.json"), 'w', encoding='utf-8') as f:
                json.dump(label_mapping, f, ensure_ascii=False, indent=2)
            
            print(f"모델 저장: {config['output_dir_path']} (Loss: {avg_loss:.4f}, Acc: {accuracy:.4f})")

#  테스트 함수
def test(config, idx2label):
    # MentalRoBERTa config 및 tokenizer 불러오기
    tokenizer = AutoTokenizer.from_pretrained(config["output_dir_path"], cache_dir=config["cache_dir_path"])
    roberta_config = AutoConfig.from_pretrained(config["output_dir_path"], cache_dir=config["cache_dir_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MentalROBERTA_SENTIMENT_CLASSIFIER.from_pretrained(config["output_dir_path"], config=roberta_config)
    model = model.to(device)

    # 평가 데이터 가져오기
    test_data = config["test_data"]
    
    # 입력 데이터 전처리
    test_input_ids, test_attention_mask = convert_data2feature(test_data, config["max_length"], tokenizer)

    # 평가 데이터를 batch 단위로 추출하기 위한 DataLoader 생성
    test_features = TensorDataset(test_input_ids, test_attention_mask)
    test_dataloader = DataLoader(test_features, sampler=SequentialSampler(test_features), batch_size=config["batch_size"])

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Softmax 함수
    softmax = nn.Softmax(dim=-1)

    # 모델 평가 모드
    model.eval()
    total_hypothesis = []

    for batch in tqdm(test_dataloader, desc="Testing"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask = batch

        with torch.no_grad():
            hypothesis = model(input_ids, attention_mask)
            hypothesis = torch.argmax(softmax(hypothesis), dim=-1)

        total_hypothesis.extend(hypothesis.cpu().numpy().tolist())

    # 예측된 라벨 매핑 적용
    for i, entry in enumerate(test_data):
        entry['predicted_label'] = idx2label[total_hypothesis[i]]

    output_dir = config["output_dir_path"]
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "test_prediction.jsonl")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in test_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"테스트 결과 저장 완료: {output_file}")
    except Exception as e:
        print(f"테스트 결과 저장 실패: {e}")

from sklearn.model_selection import train_test_split
import pandas as pd

def load_roberta_model(model_dir):
    try:
        # MentalRoBERTa config 객체 생성
        roberta_config = AutoConfig.from_pretrained(model_dir)
        setattr(roberta_config, "num_labels", 5)  # 분류 클래스 수 (PTSD 제외)
        setattr(roberta_config, "hidden_size", 768)

        # 라벨 매핑 로드
        label_mapping_path = os.path.join(model_dir, "label2idx.json")
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path, 'r', encoding='utf-8') as f:
                label_mapping = json.load(f)
            label2idx = label_mapping["label2idx"]
            idx2label = {int(k): v for k, v in label_mapping["idx2label"].items()}
        else:
            print("label2idx.json 파일이 없어 기본 매핑을 사용합니다.")
            label2idx = None
            idx2label = None

        # 커스텀 MentalRoBERTa 모델 로딩
        model = MentalROBERTA_SENTIMENT_CLASSIFIER.from_pretrained(
            model_dir,
            config=roberta_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return model, tokenizer, device, label2idx, idx2label
    except Exception as e:
        raise Exception(f"RoBERTa 모델 로딩 중 오류 발생: {str(e)}")

class MentalROBERTA_SENTIMENT_CLASSIFIER(PreTrainedModel):
    def __init__(self, config, cache_dir=None):
        super().__init__(config)
        self.roberta = AutoModel.from_pretrained(config._name_or_path, cache_dir=cache_dir)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.num_labels)

        # 명시적 가중치 초기화
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_vector)
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        logits = self.linear2(x)
        return logits

from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

if __name__ == "__main__":
    set_seed(42)

    # 데이터 로드
    print("데이터 로드 중...")
    total_data = read_jsonl("./0_datasets/merged_v2.jsonl")
    total_data = filter_data_exclude_ptsd(total_data)
    
    df = pd.DataFrame(total_data)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform([[label] for label in df['label']])
    X = df[['text']].values

    print(f"라벨 클래스: {list(mlb.classes_)}")


    # 계층적 데이터 분할 (train/valid/test)
    print("데이터 분할 중...")
    X_train_val, y_train_val, X_test, y_test = iterative_train_test_split(X, y, test_size=0.1)
    X_train, y_train, X_valid, y_valid = iterative_train_test_split(X_train_val, y_train_val, test_size=(0.1 / 0.9))
    
    
    def to_jsonl_format(X_data, y_data, mlb_instance):
        labels_tuples = mlb_instance.inverse_transform(y_data)
        return [
            {'text': text_item[0], 'label': label_tuple[0]}
            for text_item, label_tuple in zip(X_data, labels_tuples) if label_tuple
        ]

    train_data = to_jsonl_format(X_train, y_train, mlb)
    valid_data = to_jsonl_format(X_valid, y_valid, mlb)
    test_data = to_jsonl_format(X_test, y_test, mlb)


    print(f"\n데이터 분할 결과:")
    print(f"  총 데이터: {len(df)} / 훈련: {len(train_data)} / 검증: {len(valid_data)} / 테스트: {len(test_data)}")


    # 학습 설정
    label2idx, idx2label = create_label_mapping(total_data)
    num_labels = len(label2idx)
    print(f"라벨 매핑: {label2idx}")
    
    root_dir = './src/Trained_MentalRoBERTa'
    config = {
        "epoch": 1,
        "batch_size": 32,
        "max_length": 128,
        "num_labels": num_labels,
        "learning_rate": 2e-5,
        "pretrained_model_name_or_path": "mental/mental-roberta-base",
        "cache_dir_path": os.path.join(root_dir, 'MentalRoBERTa_cache'),
        "output_dir_path": os.path.join(root_dir, 'best_model'),
        "train_data": train_data,
        "valid_data": valid_data,
        "test_data": test_data,
        "label2idx": label2idx
    }

    os.makedirs(config["cache_dir_path"], exist_ok=True)
    os.makedirs(config["output_dir_path"], exist_ok=True)
    
    train(config, idx2label)
    test(config, idx2label)
    print("\n모든 작업이 완료되었습니다.")