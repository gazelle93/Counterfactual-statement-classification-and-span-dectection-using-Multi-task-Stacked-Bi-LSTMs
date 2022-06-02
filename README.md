# Overview
- Task: SemEval 2020 Task 5: Modelling Causal Reasoning in Language (Detecting Counterfactuals)
- Subtask 1: Binary classification task classifying whether the given text is counterfactual or not.
- Subtask 2: Span detection task detecting the span of the antecedent and the consequent.
- Applied architecture: Multi-task Stacked Bi-LSTMs using the grammatical feature.
- This project aims to implement the Multi-task-Stacked-Bi-LSTMs applied in detecting the span of the counterfactual statement (Subtask 2) using ELMo Word Embedding and POS tags.

# Brief description
- text_processing.py
> Output format
> - output: Tokenized result of a given text. (list)
- lstms.py
> Output format
> - output: List of tensor of attention results. (Tensor)


# Prerequisites
- argparse
- torch
- stanza
- spacy
- tqdm
- numpy
- allennlp
- pandas

# Parameters
- nlp_pipeline(str, defaults to "stanza"): NLP preprocessing pipeline.
- learning_rate(float, defaults to 1e-2): Learning rate.
- num_epochs(int, defaults to 100): The number of epochs for training.

# References
- Multi-task-Stacked-Bi-LSTMs: Sung, M., Bagherzadeh, P., & Bergler, S. (2020, December). CLaC at SemEval-2020 Task 5: Muli-task Stacked Bi-LSTMs. In Proceedings of the Fourteenth Workshop on Semantic Evaluation (pp. 445-450).
- Stanza: Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.
- Spacy: Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear (2017).
- Counterfactual Dataset: Yang, X., Obadinma, S., Zhao, H., Zhang, Q., Matwin, S., & Zhu, X. (2020). SemEval-2020 task 5: Counterfactual recognition. arXiv preprint arXiv:2008.00563. (https://github.com/arielsho/SemEval-2020-Task-5)
