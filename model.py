import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF

class SentimentModel(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(SentimentModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.cnn = nn.Conv1d(in_channels=768, out_channels=hidden_units, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_units * 2, 3)  # 3 for positive, negative, neutral
        self.crf = CRF(3, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state.permute(0, 2, 1)
        cnn_output = self.cnn(sequence_output).permute(0, 2, 1)
        lstm_output, _ = self.bilstm(cnn_output)
        lstm_output = self.dropout(lstm_output)
        emissions = self.fc(lstm_output)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.byte())
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.byte())
