![Python 3.8](https://img.shields.io/badge/python-3.8-green)

>Codes for **UniMF: A Unified Multimodal Framework for Multimodal Sentiment Analysis in Missing Modalities and Unaligned Multimodal Sequences**.

## Usage
### Clone the repository
    git clone https://github.com/gw-zhong/UniMF.git
### Download the datasets and BERT models
+ [CMU-MOSI & CMU-MOSEI (**Glove**) [align & unaligned]](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/) (which are not available now)

+ [CMU-MOSI & CMU-MOSEI (**BERT**) [align & unaligned]](https://github.com/thuiar/MMSA)

+ [MELD (**Sentiment**/**Emotion**) [align]](https://drive.google.com/drive/folders/10j3bWgAwD6i4obYoOWlxDYQQuX7DPPwa?usp=sharing) (only Glove)

+ [UR-FUNNY (**V1** & **V2**) [align]](https://github.com/ROC-HCI/UR-FUNNY) (only Glove)

Alternatively, you can download these datasets from:
- [BaiduYun Disk](https://pan.baidu.com/s/16UcDXgwmq9kxHf6ziJcChw) ```code: zpqk```

For convenience, we also provide the BERT pre-training model that we fine-tuned with:

- [pretrained_berts](https://pan.baidu.com/s/12zhRpTEx5589Bmo0OAF5cg) ```code: e7mw```

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
Or, you can run the code as normal:
 ```python
python main.py --[FLAGS]
 ```