# VLSP 2023: Comparative Opinion Mining from Vietnamese Product Reviews
Finetune multiple pre-trained Transformer-based models to solve the challenge of Comparative Opinion Mining from Vietnamese Product Reviews in the VLSP2023 shared task.

- Current baseline:
    - [ ] Huggingface Transformers: 3 separated backbones for 3 tasks
        - Pre-tokenization: [RDRsegmenter](https://github.com/datquocnguyen/RDRsegmenter): currently using [py-vncorenlp](https://github.com/vncorenlp/VnCoreNLP) implementation. (using only for PhoBERT)
        - Tokenization: 3 main tokenizers 
            - [PhoBERT tokenizer](https://huggingface.co/docs/transformers/model_doc/phobert)
            - [Bert based multilingual cased](https://huggingface.co/bert-base-multilingual-cased)
            - [NlpHUST/ner vietnamese electra base](https://huggingface.co/NlpHUST/ner-vietnamese-electra-base)
        - Main model: 
            - Task 1 & 3:
                - [PhoBERT-base](https://github.com/VinAIResearch/PhoBERT) (v2) for token classification using [Huggingface implementation](https://huggingface.co/docs/transformers/model_doc/phobert)
                - Currently do bootstraping for tasks 3. 
            - Task 2: Ensembling 0.3 * electra_output + 0.2 * phobert_output + 0.5 * multi_output 
                - [PhoBERT-base](https://github.com/VinAIResearch/PhoBERT) (v2) for token classification.
                - [Bert based multilingual cased](https://huggingface.co/bert-base-multilingual-cased)
                - [NlpHUST/ner vietnamese electra base](https://huggingface.co/NlpHUST/ner-vietnamese-electra-base)
        - Post process:
            - Relatively solve the multi labels sentence (multi quintuples for a sentence) by (1) split them to a quintuple consisting 4 list that contains all the subjects, objects, aspects, and predicates (see [function split_quintuple()](postProcess/ensembling.ipynb) and (2) generate the combination of them, which then inherit the same preference label.
            - Solve the situation when the subject and object are the same (which contains "cả hai" or "cả 2")
        - Current restriction (to be updated):
            - The final preference label are the same for all quintuple of a sentence. This make the multi label sentence output rarely correct.
            - Ensemble for task 1 and 3. 
            - Have normalized tensor when ensembling but not yet implemented. 
            - Enrich the data.
            - Review dictionaries for predicates.
            - Build a multihead model.
        - Results: E-T5-MACRO-F1: 0.130400 (see [details](Output/scores.txt))
