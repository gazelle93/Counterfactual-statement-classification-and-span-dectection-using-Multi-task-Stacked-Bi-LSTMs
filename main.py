import torch
import argparse
import random
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from text_processing import get_target
import my_onehot
from lstms import SequenceTagger
from utils import feature_to_idx

def main(args):
    cur_text = "However, he also doubts that a hacker would have much interest in the blood pressure readings you're sending to your doctor because if you are in his shoes, you'd find it difficult to make profit off that data."
    ant_span_text = "if you are in his shoes"
    con_span_text = "you'd find it difficult to make profit off that data"

    tokens, pos_span_target, sequence_span_target = get_target(cur_text, ant_span_text, con_span_text,"spacy")

    # util part
    pos_to_ix, r_pos_to_ix, tag_to_ix, r_tag_to_ix = feature_to_idx(pos_span_target, sequence_span_target)

    # One-hot Encoding
    embeddings = my_onehot.get_onehot_encoding([cur_text], cur_text, args.nlp_pipeline, args.unk_ignore)

    emb_dim = embeddings.size()[1]
    model = SequenceTagger(input_dim=emb_dim, hidden_dim=emb_dim, pos_to_ix=pos_to_ix, tag_to_ix=tag_to_ix)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)



    for epoch in range(args.num_epochs):
        model.train()

        rnd = random.random()
        returning_layer = round(rnd)

        pred = model(embeddings, returning_layer)

        # predicting POS tags
        if returning_layer == 0:
            cur_target = torch.tensor([r_pos_to_ix[x] for x in pos_span_target])
            loss = loss_function(pred, cur_target)

        else:
            cur_target = torch.tensor([r_tag_to_ix[x] for x in sequence_span_target])
            loss = loss_function(pred, cur_target)

        loss.backward()
        optimizer.step()


    model.eval()
    pred = model(embeddings, 1)
    pred = [tag_to_ix[x] for x in np.argmax(pred.detach().numpy(), axis=1)]

    print("Prediction: {}".format(pred))
    print("Actual Target: {}".format(sequence_span_target))

    a_match = 0
    c_match = 0
    for p, t in zip(pred, sequence_span_target):
        if p == t =="A":
            a_match+=1
        elif p == t =="C":
            c_match+=1

    print("Antecedent Span Match: {} / {}" .format(a_match, len([x for x in sequence_span_target if x == "A"])))
    print("Consequent Span Match: {} / {}" .format(c_match, len([x for x in sequence_span_target if x == "C"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nlp_pipeline", default="spacy", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--unk_ignore", default=True, help="Ignore unknown tokens.")
    parser.add_argument("--learning_rate", default=1e-2, help="Learning rate.")
    parser.add_argument("--num_epochs", default=100, help="Number of epochs for training.")
    args = parser.parse_args()

    main(args)
