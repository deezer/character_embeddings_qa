# Improving Quotation Attribution with Fictional Character Embeddings

This is the official repository for the EMNLP 2024 (findings) paper ["Improving Quotation Attribution with Fictional Character Embeddings"](https://aclanthology.org/2024.findings-emnlp.744.pdf). The uses [LUAR models](https://aclanthology.org/2021.emnlp-main.70/) trained on Drama plays to distinguish utterances of fictional characters, and use the resulting models to derive character representations that are further injected in a [Quotation Attribution](https://aclanthology.org/2023.acl-short.64/) model to improve the accuracy on unseen literary works.

This repository contains three subfolders:

- **UAR** that includes all the code and data used to train LUAR on drama plays.
- **quotation_attribution**, a [clone](https://github.com/Priya22/speaker-attribution-acl2023) from [BookNLP+](https://aclanthology.org/2023.acl-short.64.pdf) where we modify the original quotation attribution model. This folder contains all code to train and reproduce our quotation attribution experiments.

# Installation
Run the following commands to create an environment and install all the required packages:
```bash
python3 -m venv charemb
. ./charemb/bin/activate
pip3 install -U pip
pip3 install -r requirements.txt
```

# Running the code

Each folder has its own README file, with instructions to run code.