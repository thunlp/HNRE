# HNRE
Codes and datasets for our paper "Hierarchical Relation Extraction with Coarse-to-Fine Grained Attention"

If you use the code, please cite the following [paper](http://aclweb.org/anthology/D18-1247):

```
 @inproceedings{han2018hierarchicalRE,
   title={Hierarchical Relation Extraction with Coarse-to-Fine Grained Attention},
   author={Han, Xu and Yu, Pengfei and Liu, Zhiyuan and Sun, Maosong and Li, Peng},
   booktitle={Proceedings of EMNLP},
   year={2018}
 }
```


## Requirements

The model is implemented using tensorflow. The versions of packages used are shown below.

* tensorflow = 1.4.1
* numpy = 1.13.3
* scipy = 0.19.1

## Initialization

First unzip the `./raw_data/data.zip` and put all the files under `./raw_data`. Once the original raw text corpus data is in `./raw_data`, run

    python scripts/initial.py

## Train the model
For CNN hierarchical model,
    
    PYTHONPATH=. python scripts/train.py --model cnn_hier

For PCNN,

    PYTHONPATH=. python scripts/train.py --model pcnn_hier

## Evaluate the model

Run various evaluation by specifying `--mode` in commandline, see the paper for detailed description for these evaluation methods.

    PYTHONPATH=. python scripts/evaluate.py --mode [test method: pr, pone, ptwo, pall] --test-single --test_start_ckpt [ckpt number to be tested] --model [cnn_hier or pcnn_hier]

The logits are saved at `./outputs/logits/`. To see the PR curve, run the following command which directly `show()` the curve, and you can adjust the codes in `./scripts/show_pr.py` for saving the image as pdf file or etc. :
    
    python scripts/show_pr.py [path/to/generated .npy logits file from evaluation]

## Pretrained models

The pretrained models is already saved at `./outputs/ckpt/`. To directly evaluate on them, run the following command:

    PYTHONPATH=. python scripts/evaluate.py --mode [test method: pr, pone, ptwo, pall] --test-single --test_start_ckpt 0 --model [cnn_hier or pcnn_hier]

And PR curves can be generated same way as above.

## Baseline models

[+ATT,+ONE](https://github.com/thunlp/NRE )
[+ADV](https://github.com/jxwuyi/AtNRE)
[+SL](https://github.com/tyliupku/soft-label-RE)
Some of other baselines can be found in [other baselines](https://github.com/tyliupku/soft-label-RE/tree/master/emnlp17_plot).

