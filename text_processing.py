import stanza
import spacy

def get_nlp_pipeline(_nlp_pipeline):
    if _nlp_pipeline == "stanza":
        return stanza.Pipeline('en')
    elif _nlp_pipeline == "spacy":
        return spacy.load("en_core_web_sm")
    

def word_tokenization(_input_text, nlp, _nlp_pipeline=None, _lower=False):
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


def get_span_of_target(input_text, target_text):
    for i in range(len(input_text)):
        if input_text[i:i+len(target_text)] == target_text:
            return [x for x in range(i,i+len(target_text))]


def get_output_label(input_text, ant_span, con_span):
    label_list = ["O", "A", "C"]

    output_label = ["O" for x in range(len(input_text))]
    for idx, _ in enumerate(input_text):
        if idx in ant_span:
            output_label[idx] = "A"
        elif idx in con_span:
            output_label[idx] = "C"
    return output_label

def get_target(input_text, ant_span, con_span, nlp_pipeline):
    selected_nlp_pipeline = get_nlp_pipeline(nlp_pipeline)

    processed_cur_text, processed_cur_pos = word_tokenization(input_text, selected_nlp_pipeline, nlp_pipeline)
    processed_ant_span_text, processed_ant_pos = word_tokenization(ant_span, selected_nlp_pipeline, nlp_pipeline)
    processed_con_span_text, processed_con_pos = word_tokenization(con_span, selected_nlp_pipeline, nlp_pipeline)

    ant_target_span = get_span_of_target(processed_cur_text, processed_ant_span_text)
    con_target_span = get_span_of_target(processed_cur_text, processed_con_span_text)

    pos_span_target = processed_cur_pos
    sequence_span_target = get_output_label(processed_cur_text, ant_target_span, con_target_span)

    return processed_cur_text, pos_span_target, sequence_span_target
