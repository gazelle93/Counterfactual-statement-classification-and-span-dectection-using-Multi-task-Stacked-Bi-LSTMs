def feature_to_idx(pos_span_target, sequence_span_target):
    pos_to_ix = {}
    r_pos_to_ix = {}
    tag_to_ix = {}
    r_tag_to_ix = {}

    for idx, pos in enumerate(set(pos_span_target)):
        pos_to_ix[idx] = pos
        r_pos_to_ix[pos] = idx
    for idx, tag in enumerate(set(sequence_span_target)):
        tag_to_ix[idx] = tag
        r_tag_to_ix[tag] = idx

    return pos_to_ix, r_pos_to_ix, tag_to_ix, r_tag_to_ix
