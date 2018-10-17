# Bidirectional Retrieval Made Simple

Overview:
1. [Summary](#summary)
2. [Results](#results)
3. [Getting started](#start) 
    * [Dependencies](#depend)
    * [Download data](#data)
4. [Train new models](#train)
5. [Evaluate models](#evaluate)
6. [Citation](#citation)
7. [License](#license)

## <a name="data"></a> Training

` TODO: update scripts `

* Baseline models: `sh scripts/run.sh`
* Precomp Features: `sh scripts/run_precomp.sh`
* Precomp + DA: `sh scripts/run_precomp_da.sh`

## <a name="results"></a> Validation Results

` TODO: update results `

Results achieved using this repository (COCO-1k test set) using pre-computed features (note that we do not finetune the network in this experiment): 

| Method    | Features | R@1 | R@10| R@1 | R@10 |
| :-------: | :----: | :-------: | :-------: | :-------: | :-------: |
| RFF-net  [baseline@ICCV"17] | ResNet152 |  56.40 |  91.50 | 43.90 |  88.60 |
| `chain-v1`  (p=1, d=1024) | `resnet152_precomp` |  57.80 | 95.60 | 44.18 | 90.66 |
| `chain-v1`  (p=1, d=2048) | `resnet152_precomp` |  59.90 | 94.80 | 45.08 | 90.54 |
| `chain-v1`  (p=1, d=8192) | `resnet152_precomp` |  61.20 | 95.80 | 46.60 | 90.92 |


## <a name="start"></a> Getting Started

For getting started you will need to setup your environment and download the required data.

### <a name="depend"></a> Dependencies
We recommended to use Anaconda for the following packages.

* Python 2.7 or 3.x
* [PyTorch](http://pytorch.org/) (>0.1.12)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* TensorBoardX: `pip install tensorboardX`
* Punkt Sentence Tokenizer:
```
python
import nltk
nltk.download('punkt')
```

### <a name="data"></a> Download data

Pre-computed features: 
```bash
wget http://lsa.pucrs.br/jonatas/seam-data/irv2_precomp.tar.gz
wget http://lsa.pucrs.br/jonatas/seam-data/resnet152_precomp.tar.gz
wget http://lsa.pucrs.br/jonatas/seam-data/vocab.tar.gz
```

* The directory of the `*_precomp.tar.gz` files are referred as `$DATA_PATH`
* Extract `vocab.tar.gz` to `./vocab` directory (*required for baselines only*).

## <a name="evaluate"></a> Evaluate pre-trained models

```python
from vocab import Vocabulary
import evaluation
evaluation.evalrank("$RUN_PATH/model_best.pth.tar", data_path="$DATA_PATH", split="test")'
```

To evaluate in COCO-1cv test set, pass `fold5=True` with a model trained using 
`--data_name coco`.


## <a name="citation"></a>Citation 

If you found this code/paper useful, please cite the following papers:

```
@InProceedings{wehrmann2018cvpr,
author = {Wehrmann, Jônatas and Barros, Rodrigo C.},
title = {Bidirectional Retrieval Made Simple},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}

@article{faghri2017vse++,
  title={VSE++: Improving Visual-Semantic Embeddings with Hard Negatives},
  author={Faghri, Fartash and Fleet, David J and Kiros, Jamie Ryan and Fidler, Sanja},
  journal={arXiv preprint arXiv:1707.05612},
  year={2017}
}
```

## <a name="license"></a> License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)