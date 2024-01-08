![Python 3.8](https://img.shields.io/badge/python-3.8-green)

>Codes for **[UniMF: A Unified Multimodal Framework for Multimodal Sentiment Analysis in Missing Modalities and Unaligned Multimodal Sequences](https://ieeexplore.ieee.org/document/10339893)** （Accepted by IEEE Transactions on Multimedia）.

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

- [BaiduYun Disk](https://pan.baidu.com/s/12zhRpTEx5589Bmo0OAF5cg) ```code: e7mw```

### Preparation
First, install the required packages for your virtual environment:
 ```
pip install -r requirements.txt
 ```
Then, create (empty) folders for data, results, and pre-trained models:
 ```python
cd UniMF
 mkdir data results pre_trained_models
```
and put the downloaded data in 'data/'.
### Quick Start
To make it easier to run the code, we have provided scripts for each dataset:
- input_modalities: The input modality, which can be any of ```LAV```, ```LA```, ```LV```, ```AV```, ```L```, ```A```, ```V```.
- experiment_id: The id of the experiment, which can be set to an arbitrary integer number.
- number_of_trials: Number of trials for hyperparameter optimization.
- subdataset_name: Only MELD exists, set to ```meld_senti``` or ```meld_emo``` for MELD (Sentiment) or MELD (Emotion) respectively.

Note: If you want to run bert mode, add ```--use_bert``` and change the dataset name to ```mosi-bert``` or ```mosei-bert```.
#### MOSI
```bash
bash scripts/mosi.sh [input_mdalities] [experiment_id] [number_of_trials]
```
#### MOSEI
```bash
bash scripts/mosei.sh [input_mdalities] [experiment_id] [number_of_trials]
```
#### MELD
```bash
bash scripts/meld.sh [input_mdalities] [experiment_id] [subdataset_name] [number_of_trials]
```
#### UR-FUNNY
```bash
bash scripts/urfunny.sh [input_mdalities] [experiment_id] [number_of_trials]
```
Or, you can run the code as normal:
 ```python
python main.py --[FLAGS]
 ```
## Citation
Please cite our paper if you find that useful for your research:
 ```bibtex
@article{huan2023unimf,
   title={UniMF: A Unified Multimodal Framework for Multimodal Sentiment Analysis in Missing Modalities and Unaligned Multimodal Sequences},
   author={Huan, Ruohong and Zhong, Guowei and Chen, Peng and Liang, Ronghua},
   journal={IEEE Transactions on Multimedia},
   year={2023},
   publisher={IEEE}
}
 ```
## Contact
If you have any question, feel free to contact me through [guoweizhong@zjut.edu.cn](guoweizhong@zjut.edu.cn).