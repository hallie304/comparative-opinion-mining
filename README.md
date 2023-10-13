# VLSP 2023: Comparative Opinion Mining from Vietnamese Product Reviews
Finetune pre-trained Transformer-based models for a VLSP2023 shared task - Comparative Opinion Mining from Vietnamese Product Reviews.

- Current baseline:
    - [ ] PhoBERT: Huggingface Transformers:
        - Pre-tokenization: [RDRsegmenter](https://github.com/datquocnguyen/RDRsegmenter): currently using [py-vncorenlp](https://github.com/vncorenlp/VnCoreNLP) implementation.
        - Tokenization: [PhoBERT tokenizer](https://huggingface.co/docs/transformers/model_doc/phobert)
        - Main model: [PhoBERT-base](https://github.com/VinAIResearch/PhoBERT) (v1) for token classification using [Huggingface implementation](https://huggingface.co/docs/transformers/model_doc/phobert)
        - Current restriction:
            - Only handle 1 label per 1 input
            - Loss function can be further improved
            - Imbalance dataset handling
            - Review dictionaries for labels
            - Make sure the metrics work properly
            - Post-process
            - Retrain and upload results
            - Comment code
        - Results: TBA
            