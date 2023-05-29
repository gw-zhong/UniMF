![Python 3.8](https://img.shields.io/badge/python-3.8-green)

>Codes for **UniMF: A Unified Multimodal Framework for Multimodal Sentiment Analysis in Missing Modalities and Unaligned Multimodal Sequences**.

## Usage
### Clone the repository
    git clone https://gitee.com/zhongguowei-zjut/unimf.git
### Download the datasets and BERT models
+ [CMU-MOSI & CMU-MOSEI (**Glove**) [align & unaligned]](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/) (which are not available now)

+ [CMU-MOSI & CMU-MOSEI (**BERT**) [align & unaligned]](https://github.com/thuiar/MMSA)

+ [MELD (**Sentiment**/**Emotion**) [align]](https://github.com/deepsuperviser/CTFN) (only Glove)

+ [UR-FUNNY (**V1** & **V2**) [align]](https://github.com/ROC-HCI/UR-FUNNY) (only Glove)

For convenience, we also provide the BERT pre-training model that we fine-tuned with:

[bert_cn]()

[bert_en]()
### Preparation
Create (empty) folders for data, results, and pre-trained models:
 ```python
cd unimf
 mkdir data results pre_trained_models
```
and put the downloaded data in 'data/'.
### Quick Start
To make it easier to run the code, we have provided scripts for each dataset:
#### MOSI
```bash
bash scripts/mosi.sh [input_mdalities] [experiment_id]
```
#### MOSEI
```bash
bash scripts/mosei.sh [input_mdalities] [experiment_id]
```
#### MELD
```bash
bash scripts/meld.sh [input_mdalities] [experiment_id] [subdataset_name]
```
#### UR-FUNNY
```bash
bash scripts/urfunny.sh [input_mdalities] [experiment_id]
```
### Run the code
 ```python
python main.py --[FLAGS]
 ```