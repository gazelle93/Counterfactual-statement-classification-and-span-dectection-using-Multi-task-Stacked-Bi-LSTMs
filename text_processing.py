import stanza
import spacy
from allennlp.modules.elmo import Elmo, batch_to_ids

def get_model():
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo = Elmo(options_file=options_file,
                weight_file=weight_file,
                num_output_representations=2,
                dropout=0)

    return elmo

def get_wordembedding(embedding_model, datasamples):
    embeddings = []
    for datasample in datasamples:
        elmo_input = batch_to_ids([datasample])
        embeddings.append(embedding_model(elmo_input)['elmo_representations'][0])
    return embeddings




def get_nlp_pipeline(_nlp_pipeline, device):
    if _nlp_pipeline == "stanza":
        if device:
            return stanza.Pipeline('en', use_gpu= True)
        else:
            return stanza.Pipeline('en')
    elif _nlp_pipeline == "spacy":
        if device:
            spacy.prefer_gpu()
        return spacy.load("en_core_web_sm")
    

def word_tokenization(_input_text, nlp, _nlp_pipeline=None, _lower=False):
    if _input_text != "{}":
        if _lower == True:
            _input_text = _input_text.lower()

        input_tk_list = []
        input_pos_list = []

        if _nlp_pipeline == None:
            return _input_text.split()

        elif _nlp_pipeline == "stanza":
            text = nlp(_input_text)

            for sen in text.sentences:
                for tk in sen.tokens:
                    tk_info_dict = tk.to_dict()[0]
                    cur_tk = tk_info_dict["text"]
                    input_tk_list.append(cur_tk)
                    input_pos_list.append(tk_info_dict["xpos"])
            return input_tk_list, input_pos_list

        elif _nlp_pipeline == "spacy":
            text = nlp(_input_text)

            for tk_idx, tk in enumerate(text):
                cur_tk = tk.text
                input_tk_list.append(cur_tk)
                input_pos_list.append(tk.pos_)

            return input_tk_list, input_pos_list
    return [], []


def get_span_of_target(input_text, target_text):
    if target_text != []:
        for i in range(len(input_text)):
            if input_text[i:i+len(target_text)] == target_text:
                return [x for x in range(i,i+len(target_text))]
        return []
    else:
        return []

def get_output_label(input_text, ant_span, con_span):
    output_label = ["O" for x in range(len(input_text))]

    for idx, _ in enumerate(input_text):
        if idx in ant_span:
            output_label[idx] = "A"
        elif idx in con_span:
            output_label[idx] = "C"
    return output_label

def get_target(input_text, ant_span, con_span, selected_nlp_pipeline, nlp_pipeline):

    processed_cur_text, processed_cur_pos = word_tokenization(input_text, selected_nlp_pipeline, nlp_pipeline)
    processed_ant_span_text, processed_ant_pos = word_tokenization(ant_span, selected_nlp_pipeline, nlp_pipeline)
    processed_con_span_text, processed_con_pos = word_tokenization(con_span, selected_nlp_pipeline, nlp_pipeline)

    ant_target_span = get_span_of_target(processed_cur_text, processed_ant_span_text)
    con_target_span = get_span_of_target(processed_cur_text, processed_con_span_text)

    pos_span_target = processed_cur_pos
    sequence_span_target = get_output_label(processed_cur_text, ant_target_span, con_target_span)

    return processed_cur_text, pos_span_target, sequence_span_target
