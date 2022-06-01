import torch.nn as nn

class LSTMTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bidirectional=True):
        super().__init__()
        self.lstmlayer = nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=0.01)
        self.activation = nn.ReLU()
        self.tagger = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, input_embedding):
        output_states, (hidden_states, cell_states) = self.lstmlayer(input_embedding)
        output = self.dropout(output_states)
        output = self.activation(output)
        output = self.tagger(output)
        return output, (output_states, hidden_states, cell_states)



class SequenceTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, pos_to_ix, tag_to_ix):
        super().__init__()

        self.pos_tagger = LSTMTagger(input_dim, hidden_dim//2, len(pos_to_ix), bidirectional=True)
        self.ac_tagger = LSTMTagger(input_dim, hidden_dim//2, len(tag_to_ix), bidirectional=True)

        self.dropout = nn.Dropout(p=0.01)


    def forward(self, input_embedding, returning_layer):

        predicted_pos, lstm_tuple = self.pos_tagger(input_embedding)

        if returning_layer==0:
            return predicted_pos

        hidden_states = self.dropout(lstm_tuple[0])
        predicted_tags, _ = self.ac_tagger(hidden_states)

        return predicted_tags
