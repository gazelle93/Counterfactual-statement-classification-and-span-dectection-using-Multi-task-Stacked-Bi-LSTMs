import torch
import argparse
import random
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from text_processing import get_model, get_wordembedding
from lstms import SequenceTagger
from utils import get_features, load_datasamples, get_processed_datasamples, feature_to_idx, get_evaluation_result

def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("---Running on GPU.")
    else:
        device = torch.device('cpu')
        print("---Running on CPU.")

    # load & preprocess dataset
    print("---Initiating Dataset Loading & Pre-processing.")
    train_dataset, test_dataset = load_datasamples(args.subtask)
    train_dataset = get_processed_datasamples(train_dataset[:10], args.subtask)
    test_dataset = get_processed_datasamples(test_dataset[:10], args.subtask)

    train_datasamples = get_features(train_dataset, args.nlp_pipeline, args.subtask, device)
    test_datasamples = get_features(test_dataset, args.nlp_pipeline, args.subtask, device)

    pos_datasample = np.concatenate((train_datasamples[:,1], test_datasamples[:,1]))
    pos_to_ix, r_pos_to_ix, tag_to_ix, r_tag_to_ix = feature_to_idx(pos_datasample, args.subtask)
    print("---Done Dataset Loading & Pre-processing.")

    # get ELMo word embedding model
    print("---Initiating Word embedding (ELMo).")
    embedding_model = get_model()
    train_set = get_wordembedding(embedding_model, train_datasamples[:,0])
    test_set = get_wordembedding(embedding_model, test_datasamples[:,0])
    print("---Done Word embedding (ELMo).")

    # model
    emb_dim = train_set[0].size()[2]
    model = SequenceTagger(input_dim=emb_dim, hidden_dim=emb_dim, pos_to_ix=pos_to_ix, tag_to_ix=tag_to_ix).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("---Initiating training process.")
    # training
    for epoch in range(args.num_epochs):
        model.train()

        for idx in tqdm(range(len(train_set))):
            model.zero_grad()
            rnd = random.random()
            returning_layer = round(rnd)

            pred = model(train_set[idx], returning_layer)

            # predicting POS tags
            if returning_layer == 0:
                pred = pred.reshape(-1, len(pos_to_ix))
                cur_target = torch.tensor([r_pos_to_ix[x] for x in train_datasamples[idx,1]])
                loss = loss_function(pred, cur_target)

            else:
                pred = pred.reshape(-1, len(tag_to_ix))
                cur_target = torch.tensor([r_tag_to_ix[x] for x in train_datasamples[idx,2]])
                loss = loss_function(pred, cur_target)

            loss.backward(retain_graph=True)
            optimizer.step()


    # evaluation
    model.eval()

    prediction_list = []
    gold_list = []
    for idx in range(len(test_set)):
        pred = model(test_set[idx], 1)
        pred = pred.reshape(-1, len(tag_to_ix))
        pred = [tag_to_ix [x] for x in np.argmax(pred.detach().numpy(), axis=1)]

        prediction_list.append(pred)
        gold_list.append(test_datasamples[idx,2])

    get_evaluation_result(gold_list, prediction_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--subtask", default="2", help="The selection of subtask (1 or 2).")
    parser.add_argument("--nlp_pipeline", default="spacy", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--learning_rate", default=1e-3, help="Learning rate.")
    parser.add_argument("--num_epochs", default=5, help="Number of epochs for training.")
    args = parser.parse_args()

    main(args)
