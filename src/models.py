"""
Recurrent Neural Network Architectures for Sentiment Analysis.
Team 2 - Task 2A: Sentiment Analysis of Code Mixed Telugu and English Reviews.
Model Lead: 23265A6606
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentRNN(nn.Module):
    """
    Unified Recurrent Architecture supporting:
    - Simple RNN (Elman RNN)
    - LSTM (Long Short-Term Memory)
    - GRU (Gated Recurrent Unit)
    - Unidirectional and Bidirectional variants
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3, 
                 cell_type='gru', bidirectional=True, dropout=0.3, pad_idx=0):
        super(SentimentRNN, self).__init__()
        self.cell_type = cell_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        if self.cell_type == 'rnn':
            self.rnn = nn.RNN(
                embed_dim, hidden_dim, batch_first=True, 
                bidirectional=bidirectional, nonlinearity='tanh'
            )
        elif self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                embed_dim, hidden_dim, batch_first=True, 
                bidirectional=bidirectional
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                embed_dim, hidden_dim, batch_first=True, 
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}. Choose 'rnn', 'lstm', or 'gru'.")
            
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x)) # [batch_size, seq_len, embed_dim]
        
        if self.cell_type == 'lstm':
            out, (hn, cn) = self.rnn(embedded)
        else:
            out, hn = self.rnn(embedded)
            
        # If bidirectional, concatenate forward and backward final hidden states
        if self.bidirectional:
            # hn shape: [num_layers * num_directions, batch_size, hidden_dim]
            # hn[-2] is forward last hidden state, hn[-1] is backward last hidden state
            final_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            final_hidden = hn[-1]
            
        out = self.dropout(final_hidden)
        logits = self.fc(out)
        return logits

