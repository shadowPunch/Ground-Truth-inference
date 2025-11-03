import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        energy = torch.tanh(self.Wa(decoder_hidden) + self.Ua(encoder_outputs))
        attention_scores = self.Va(energy).squeeze(2)
        return F.softmax(attention_scores, dim=1)

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, input_token, hidden_state, cell_state, encoder_outputs):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        top_hidden = hidden_state[-1].unsqueeze(1)
        attn_weights = self.attention(top_hidden, encoder_outputs)
        attn_weights_expanded = attn_weights.unsqueeze(1)
        context_vector = torch.bmm(attn_weights_expanded, encoder_outputs)
        lstm_input = torch.cat((embedded, context_vector), dim=2)
        output, (hidden_state, cell_state) = self.lstm(lstm_input, (hidden_state, cell_state))
        prediction = self.output_proj(output.squeeze(1))
        return prediction, hidden_state, cell_state, attn_weights

class BiasNeutralizationModel(nn.Module):
    def __init__(self, encoder: BertModel, decoder: LSTMDecoder, vocab_size: int):
        super(BiasNeutralizationModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask, decoder_input_ids, target_labels=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled_output = encoder_outputs[:, 0, :]
        hidden_state = pooled_output.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        cell_state = torch.zeros_like(hidden_state, device=encoder_outputs.device)
        batch_size = input_ids.size(0)
        target_len = decoder_input_ids.size(1)
        outputs = torch.zeros(batch_size, target_len, self.vocab_size, device=encoder_outputs.device)
        input_token = decoder_input_ids[:, 0]
        for t in range(target_len):
            prediction, hidden_state, cell_state, _ = self.decoder(
                input_token, hidden_state, cell_state, encoder_outputs
            )
            outputs[:, t] = prediction
            if target_labels is not None:
                if t + 1 < target_len:
                    input_token = decoder_input_ids[:, t + 1]
                else:
                    break
            else:
                input_token = prediction.argmax(1)
        return outputs

    def generate(self, encoder_outputs, tokenizer, max_len=50):
        self.eval()
        batch_size = encoder_outputs.size(0)
        pooled_output = encoder_outputs[:, 0, :]
        hidden_state = pooled_output.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        cell_state = torch.zeros_like(hidden_state, device=encoder_outputs.device)
        input_token = torch.full((batch_size,), tokenizer.cls_token_id, dtype=torch.long, device=encoder_outputs.device)
        output_sequences = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long, device=encoder_outputs.device)
        output_sequences[:, 0] = input_token
        with torch.no_grad():
            for t in range(1, max_len):
                prediction, hidden_state, cell_state, _ = self.decoder(
                    input_token, hidden_state, cell_state, encoder_outputs
                )
                top1 = prediction.argmax(1)
                input_token = top1
                output_sequences[:, t] = top1
        decoded_sentences = []
        for i in range(batch_size):
            ids = output_sequences[i].tolist()
            try:
                eos_index = ids.index(tokenizer.sep_token_id)
                ids = ids[1:eos_index]
            except ValueError:
                ids = ids[1:]
            sentence = tokenizer.decode(ids, skip_special_tokens=True)
            decoded_sentences.append(sentence)
        return decoded_sentences
