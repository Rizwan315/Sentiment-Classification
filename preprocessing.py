import pandas as pd
from transformers import RobertaTokenizer


file_path = os.path.join(extracted_folder, 'reviews.csv')  
data = pd.read_csv(file_path)

# Inspect the data
print(data.head())

# Preprocess data
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_seq_length = 200

def tokenize_and_pad(text, max_seq_length):
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_seq_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoded_dict['input_ids'], encoded_dict['attention_mask']

data['input_ids'], data['attention_mask'] = zip(*data['text_column'].apply(lambda x: tokenize_and_pad(x, max_seq_length)))
