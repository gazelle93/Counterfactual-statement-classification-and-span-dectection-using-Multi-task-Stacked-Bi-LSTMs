import torch
import torch.nn as nn

class LSTMTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bidirectional=True):
        super().__init__()
        self.lstmlayer = nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=0.05)
        self.activation = nn.ReLU()
        self.tagger = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, input_embedding):
        output_states, (hidden_states, cell_states) = self.lstmlayer(input_embedding)
        output = self.dropout(output_states)
        output = self.activation(output)
        output = self.tagger(output)
        return output, (output_states, hidden_states, cell_states)



class SequenceTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, pos_to_ix, tag_to_ix, subtask):
        super().__init__()
        self.subtask = subtask
        self.pos_tagger = LSTMTagger(input_dim, hidden_dim//2, len(pos_to_ix), bidirectional=True)
        self.ac_tagger = LSTMTagger(input_dim, hidden_dim//2, len(tag_to_ix), bidirectional=True)

        self.dropout = nn.Dropout(p=0.05)

        if subtask == "1":
            self.clf_layer = nn.Linear(len(tag_to_ix)*2, len(tag_to_ix))



    def forward(self, input_embedding, returning_layer):

        predicted_pos, lstm_tuple = self.pos_tagger(input_embedding)

        if returning_layer==0:
            return predicted_pos

        hidden_states = self.dropout(lstm_tuple[0])
        predicted_tags, _ = self.ac_tagger(hidden_states)

        if self.subtask == "1":
            start_emb = predicted_tags[:,0,:]
            end_emb = predicted_tags[:,-1,:]

            return self.clf_layer(torch.concat((start_emb, end_emb), dim=1))
        else:
            return predicted_tags
