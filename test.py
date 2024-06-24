import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import joblib
import sys

# 모델 및 토크나이저 로드
model_path = 'best_model.pt'
model_name = 'monologg/koelectra-base-v3-discriminator'
tokenizer = ElectraTokenizer.from_pretrained(model_name)
model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=33)  # num_labels는 학습 시 설정과 동일해야 합니다.
model.load_state_dict(torch.load(model_path))
model.eval()

# LabelEncoder 로드
label_encoder = joblib.load('label_encoder.joblib')

def predict_category(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_category = label_encoder.inverse_transform([predicted_class_id])[0]

    return predicted_category

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'Your text here'")
        sys.exit(1)

    text = sys.argv[1]
    category = predict_category(text)
    print(f"The predicted category is: {category}")
