import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import config
import random

class Encoder(nn.Module):
    """BERT Encoder Wrapper"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_MODEL_NAME)

    def forward(self, input_ids, attention_mask):
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # last_hidden_state: [batch_size, seq_len, hidden_dim]
        # pooler_output: [batch_size, hidden_dim]
        return outputs.last_hidden_state, outputs.pooler_output

class Attention(nn.Module):
    """Bahdanau (Additive) Attention"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim) # For decoder hidden state
        self.Ua = nn.Linear(hidden_dim, hidden_dim) # For encoder outputs
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        # decoder_hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        # mask: [batch_size, seq_len] (source mask)

        # Add time dimension to decoder_hidden
        # decoder_hidden_expanded: [batch_size, 1, hidden_dim]
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1)

        # Calculate energy scores
        # Wa(decoder_hidden) -> [batch_size, 1, hidden_dim]
        # Ua(encoder_outputs) -> [batch_size, seq_len, hidden_dim]
        # energy: [batch_size, seq_len, hidden_dim]
        energy = torch.tanh(self.Wa(decoder_hidden_expanded) + self.Ua(encoder_outputs))
        
        # attention_scores: [batch_size, seq_len, 1] -> [batch_size, seq_len]
        attention_scores = self.Va(energy).squeeze(2)

        # Apply mask (set masked values to -infinity)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # attention_weights: [batch_size, seq_len]
        attention_weights = F.softmax(attention_scores, dim=1)

        return attention_weights

class Decoder(nn.Module):
    """Attentional LSTM Decoder"""
    def __init__(self, vocab_size, hidden_dim, lstm_layers, dropout):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attention = Attention(hidden_dim)
        
        # LSTM input: embedded token + context vector
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True, # Note: We process one token at a time, but batch_first is good practice
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, decoder_hidden, decoder_cell, encoder_outputs, source_mask):
        # input_token: [batch_size] (current token ID)
        # decoder_hidden: [lstm_layers, batch_size, hidden_dim]
        # decoder_cell: [lstm_layers, batch_size, hidden_dim]
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        # source_mask: [batch_size, seq_len]
        
        # 1. Embed input token
        # input_token: [batch_size] -> [batch_size, 1]
        input_token = input_token.unsqueeze(1)
        # embedded: [batch_size, 1, hidden_dim]
        embedded = self.dropout(self.embedding(input_token))

        # 2. Calculate attention
        # Use the top LSTM layer's hidden state for attention
        # attn_weights: [batch_size, seq_len]
        attn_weights = self.attention(decoder_hidden[-1], encoder_outputs, source_mask)
        
        # 3. Calculate context vector
        # attn_weights: [batch_size, 1, seq_len]
        attn_weights_expanded = attn_weights.unsqueeze(1)
        # context_vector: [batch_size, 1, hidden_dim]
        context_vector = torch.bmm(attn_weights_expanded, encoder_outputs)

        # 4. Concatenate embedding and context
        # rnn_input: [batch_size, 1, hidden_dim * 2]
        rnn_input = torch.cat((embedded, context_vector), dim=2)

        # 5. Pass through LSTM
        # output: [batch_size, 1, hidden_dim]
        # (decoder_hidden, decoder_cell): ([layers, B, H], [layers, B, H])
        output, (decoder_hidden, decoder_cell) = self.lstm(
            rnn_input, (decoder_hidden, decoder_cell)
        )

        # 6. Generate prediction
        # prediction: [batch_size, 1, vocab_size] -> [batch_size, vocab_size]
        prediction = self.out(output.squeeze(1))

        return prediction, decoder_hidden, decoder_cell, attn_weights

class ConcurrentModel(nn.Module):
    """Main CONCURRENT Model"""
    def __init__(self, encoder, decoder, tokenizer, device):
        super(ConcurrentModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = tokenizer.vocab_size
        
        # Per the paper: project mean BERT output to init decoder state
        self.init_hidden = nn.Linear(config.HIDDEN_DIM, config.LSTM_LAYERS * config.HIDDEN_DIM)
        self.init_cell = nn.Linear(config.HIDDEN_DIM, config.LSTM_LAYERS * config.HIDDEN_DIM)

    def _init_decoder_state(self, encoder_outputs, pooler_output):
        """
        Initializes decoder hidden and cell states from BERT outputs.
        Paper uses mean of last hidden state. Let's use that.
        """
        # mean_encoder_output: [batch_size, hidden_dim]
        mean_encoder_output = encoder_outputs.mean(dim=1)
        
        # Reshape to [lstm_layers, batch_size, hidden_dim]
        batch_size = mean_encoder_output.size(0)
        
        initial_hidden = torch.tanh(self.init_hidden(mean_encoder_output))
        initial_hidden = initial_hidden.view(
            config.LSTM_LAYERS, batch_size, config.HIDDEN_DIM
        ).contiguous()
        
        initial_cell = torch.tanh(self.init_cell(mean_encoder_output))
        initial_cell = initial_cell.view(
            config.LSTM_LAYERS, batch_size, config.HIDDEN_DIM
        ).contiguous()

        return initial_hidden, initial_cell

    def forward(self, source_ids, source_mask, target_ids, teacher_forcing_ratio):
        # source_ids, source_mask: [batch_size, source_len]
        # target_ids: [batch_size, target_len]
        
        batch_size = source_ids.size(0)
        target_len = target_ids.size(1)

        # Store decoder outputs
        # outputs: [batch_size, target_len, vocab_size]
        outputs = torch.zeros(batch_size, target_len, self.vocab_size).to(self.device)
        
        # 1. Pass source through encoder
        encoder_outputs, pooler_output = self.encoder(source_ids, source_mask)
        
        # 2. Initialize decoder state
        hidden, cell = self._init_decoder_state(encoder_outputs, pooler_output)

        # 3. Decoder loop
        # Start with <CLS> token (or <SOS> if you have one)
        input_token = target_ids[:, 0]
        
        for t in range(1, target_len): # Skip <CLS>
            prediction, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs, source_mask
            )
            
            # Store prediction
            outputs[:, t, :] = prediction
            
            # Decide whether to use teacher forcing
            use_teacher_force = random.random() < teacher_forcing_ratio
            
            if use_teacher_force:
                input_token = target_ids[:, t]
            else:
                # Use the model's own prediction
                top1 = prediction.argmax(1)
                input_token = top1
                
        return outputs

    def translate(self, source_text, max_len=config.MAX_LEN):
        """Inference method for translating a single sentence"""
        self.eval() # Set to evaluation mode
        
        # Tokenize source text
        inputs = self.tokenizer(
            source_text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        source_ids = inputs['input_ids'].to(self.device)
        source_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            encoder_outputs, pooler_output = self.encoder(source_ids, source_mask)
            hidden, cell = self._init_decoder_state(encoder_outputs, pooler_output)
            
            # Start with <CLS> token
            input_token = torch.tensor([self.tokenizer.cls_token_id], device=self.device)
            
            output_tokens = []
            
            for _ in range(max_len):
                prediction, hidden, cell, _ = self.decoder(
                    input_token, hidden, cell, encoder_outputs, source_mask
                )
                
                # Get the most likely token ID
                top1 = prediction.argmax(1)
                input_token = top1
                
                # Stop if <SEP> token is generated
                if top1.item() == self.tokenizer.sep_token_id:
                    break
                
                output_tokens.append(top1.item())
                
        # Decode the generated token IDs
        return self.tokenizer.decode(output_tokens, skip_special_tokens=True)

