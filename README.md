# BERT-Based Chatbot for Question Answering

This project demonstrates how to fine-tune a BERT model for a sequence classification task to build a chatbot capable of answering questions. The script covers data preprocessing, model training, and creating an interface for interacting with the fine-tuned model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Inference](#model-inference)
- [Performance Metrics](#performance-metrics)
- [Examples of Conversations](#examples-of-conversations)
- [Accessing the Chatbot](#accessing-the-chatbot)

## Installation

First, you need to install the required libraries. Use the following commands to set up your environment:

```sh
pip install pandas torch transformers
```

## Usage

### Step 1: Create the Dataset

We can get our dataset on kaagle

### Step 2: Preprocess Data

We clean the text and tokenize it using BERT's tokenizer:

```python
import re
from transformers import BertTokenizer

# Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['question'] = df['question'].apply(clean_text)
df['answer'] = df['answer'].apply(clean_text)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 512

df['question_tokens'] = df['question'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=max_len, truncation=True, padding='max_length'))
df['answer_tokens'] = df['answer'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=max_len, truncation=True, padding='max_length'))
```

### Step 3: Prepare Input Tensors for BERT

Create a dataset and dataloader:

```python
from torch.utils.data import Dataset, DataLoader
import torch

class ChatbotDataset(Dataset):
    def __init__(self, questions, answers):
        self.questions = questions
        self.answers = answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        return torch.tensor(question), torch.tensor(answer)

questions = df['question_tokens'].tolist()
answers = df['answer_tokens'].tolist()

dataset = ChatbotDataset(questions, answers)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

### Step 4: Fine-tune the BERT Model

Fine-tune the BERT model for sequence classification:

```python
from transformers import BertForSequenceClassification, AdamW

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(3):  # Adjust number of epochs as needed
    for batch in dataloader:
        questions, answers = batch
        optimizer.zero_grad()
        outputs = model(input_ids=questions, labels=answers)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Loss {loss.item()}')

# Save the fine-tuned model
model.save_pretrained('finetuned_bert_model')
```

### Step 5: Build an Interface for the Model

Create a function to interact with the fine-tuned model:

```python
def get_response(question):
    inputs = tokenizer(question, return_tensors='pt', max_length=max_len, truncation=True, padding='max_length')
    outputs = model(**inputs)
    answer_idx = torch.argmax(outputs.logits, dim=1).item()
    return answer_idx

# Example usage
question = "What are the symptoms of COVID-19?"
response = get_response(question)
print(response)
```

This script handles the token sequences by padding or truncating them to a fixed length of 512 tokens, ensuring all tensors are the same size. This prevents the runtime error you encountered. Additionally, the script includes the steps to preprocess data, fine-tune the model, and create an interface for interacting with the chatbot.

## Dataset

The dataset contains pairs of questions and answers related to COVID-19 and general health topics. Each question and answer pair is preprocessed and tokenized for input into the BERT model.

**Link to the dataset:** [Dataset Link](https://github.com/Freedisch/ml-formativeChatbot/blob/main/Covid-QA%20FAQ.csv)

## Model Training

The BERT model is fine-tuned for sequence classification. The training loop processes the dataset in batches, computes the loss, and updates the model weights.

## Model Inference

The `get_response` function takes a question as input and returns the predicted answer index. This function uses the fine-tuned BERT model to perform inference.

## Performance Metrics

Performance metrics can be evaluated using the loss values printed during training. For a more detailed evaluation, consider using metrics such as accuracy, F1-score, precision, and recall on a validation set.

## Examples of Conversations

**Question:** What are the symptoms of COVID-19?
**Response:** Common symptoms include fever, dry cough, and tiredness. Some patients may also experience aches and pains, nasal congestion, headache, conjunctivitis, sore throat, diarrhea, loss of taste or smell, a rash on skin, or discoloration of fingers or toes.

**Question:** How can I protect myself from COVID-19?
**Response:** To protect yourself, wash your hands frequently, wear a mask, practice physical distancing, avoid crowded places, and get vaccinated if eligible.

**Question:** What is diabetes?
**Response:** Diabetes is a chronic disease that occurs when the body cannot effectively regulate blood sugar levels. It is classified into type 1 and type 2 diabetes.

## Accessing the Chatbot

**Link to access the chatbot:** `docker pull freedisch/ml_pipeline:latest` (Need: Linux\ARM64 machine)

## Conclusion

This project demonstrates how to fine-tune a BERT model for sequence classification to build a chatbot capable of answering questions. The script covers data preprocessing, model training, and creating an interface for interacting with the fine-tuned model.
