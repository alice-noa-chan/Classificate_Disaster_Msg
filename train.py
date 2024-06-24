import pandas as pd
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer, ElectraForSequenceClassification, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import joblib
import os

# 데이터 로드
data = pd.read_csv('data.tsv', sep='\t')

# 'MSG_CN' 컬럼과 'DSSTR_SE_NM' 컬럼 추출
texts = data['MSG_CN'].tolist()
labels = data['DSSTR_SE_NM'].tolist()

# 카테고리 인코딩
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
label_map = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

# LabelEncoder 저장
joblib.dump(label_encoder, 'label_encoder.joblib')

# 데이터 증강 함수 정의 (RD, RS 적용)
def random_deletion(sentence, p=0.1):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if len(new_words) == 0:
        return random.choice(words)
    else:
        return ' '.join(new_words)

def random_swap(sentence, n=1):
    words = sentence.split()
    length = len(words)
    if length < 2:
        return sentence
    for _ in range(n):
        idx1, idx2 = random.sample(range(length), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def augment_text(text, num_aug=2):
    augmented_texts = [random_deletion(text) for _ in range(num_aug)] + [random_swap(text) for _ in range(num_aug)]
    return augmented_texts

# 카테고리별 데이터 증강
category_counts = Counter(labels)
max_count = max(category_counts.values())

augmented_texts = []
augmented_labels = []

for label, count in category_counts.items():
    texts_in_category = [text for text, lbl in zip(texts, labels) if lbl == label]
    augmented_texts.extend(texts_in_category)
    augmented_labels.extend([label] * len(texts_in_category))
    num_to_augment = max_count - count
    while num_to_augment > 0:
        for text in texts_in_category:
            if num_to_augment <= 0:
                break
            for aug_text in augment_text(text):
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
                num_to_augment -= 1

# 데이터셋 섞기
combined = list(zip(augmented_texts, augmented_labels))
random.shuffle(combined)
augmented_texts[:], augmented_labels[:] = zip(*combined)

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# KoElectra 토크나이저 및 모델 불러오기
tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
model = ElectraForSequenceClassification.from_pretrained('monologg/koelectra-base-v3-discriminator', num_labels=len(set(labels)))

# 데이터셋 생성
MAX_LEN = 128
dataset = CustomDataset(augmented_texts, augmented_labels, tokenizer, MAX_LEN)

# 데이터셋 분할
train_size = 0.7
val_size = 0.2
test_size = 0.1
train_texts, temp_texts, train_labels, temp_labels = train_test_split(augmented_texts, augmented_labels, test_size=(1-train_size), stratify=augmented_labels)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=(test_size/(val_size+test_size)), stratify=temp_labels)

train_dataset = CustomDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, MAX_LEN)
test_dataset = CustomDataset(test_texts, test_labels, tokenizer, MAX_LEN)

# 데이터 로더 생성
BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 훈련 설정
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # L2 정규화 적용
total_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Early Stopping 설정
class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model, model_path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_path):
        torch.save(model.state_dict(), model_path)
        self.val_loss_min = val_loss

early_stopping = EarlyStopping(patience=3, delta=0)
model_path = 'best_model.pt'

# 모델 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_true = []
    train_pred = []
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).flatten()
        train_true.extend(labels.cpu().numpy())
        train_pred.extend(preds.cpu().numpy())
        
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = accuracy_score(train_true, train_pred)
    train_f1 = f1_score(train_true, train_pred, average='weighted')

    model.eval()
    val_loss = 0
    val_true = []
    val_pred = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).flatten()
            val_true.extend(labels.cpu().numpy())
            val_pred.extend(preds.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_true, val_pred)
    val_f1 = f1_score(val_true, val_pred, average='weighted')

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print(f'Train Loss: {avg_train_loss}, Train Accuracy: {train_accuracy}, Train F1 Score: {train_f1}')
    print(f'Val Loss: {avg_val_loss}, Val Accuracy: {val_accuracy}, Val F1 Score: {val_f1}')

    early_stopping(avg_val_loss, model, model_path)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# 모델 파일 저장
torch.save(model.state_dict(), 'final_model.pt')
print("Model saved!")

# 테스트
model.load_state_dict(torch.load(model_path))
model.eval()
test_loss = 0
test_true = []
test_pred = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        test_loss += loss.item()
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).flatten()
        test_true.extend(labels.cpu().numpy())
        test_pred.extend(preds.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
test_accuracy = accuracy_score(test_true, test_pred)
test_f1 = f1_score(test_true, test_pred, average='weighted')

print(f'Test Loss: {avg_test_loss}, Test Accuracy: {test_accuracy}, Test F1 Score: {test_f1}')
