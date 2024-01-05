# Standard Images Bridge the Modality Gap between Casual Handwritten Images and Standard LaTeX Text for HMER Task

>*Beichen Zhang, *Lifeng Qiao, *Weiji Xie, *Mingyang Feng

## Abstract

<p align="justify">
As a multi-modal task, a main challenge of Handwritten Mathematical Expression Recognition (HMER) task lies in the huge modality gap between the casual handwritten image and the standard LaTeX text in both the information format and content of the information. The input information is casually formatted and is indicated in 2D image while the output information has strict grammar restriction and is expressed in 1D text. This modality gap increases the training difficulty and thus affects the performance of current methods which only train the model with handwritten image and LaTeX text. To this end, we introduce the use of standard images, the standard preview image of a particular LaTeX text, to bridge the modality gap. Specifically, we propose a three-stage training pipeline, where the standard image acts as an intermediary, facilitating the pre-training of the text decoder and aligning the encoder for handwritten images across various training phases.

Abundant experiments have shown that by incorporating standard images into the training pipeline, we observe significant performance improvements over our baseline with fewer training epochs for different datasets.

Moreover, we also propose an adaptive batch sampler to accelerate training speed by allowing larger batch size while avoiding cuda OOM problem.
</p>


## Datasets

Download the CROHME dataset from [BaiduYun](https://pan.baidu.com/s/1qUVQLZh5aPT6d7-m6il6Rg) (downloading code: 1234) and put it in ```datasets/```.

The HME100K dataset can be download from the official website [HME100K](https://ai.100tal.com/dataset).

## Training

Check the config file ```config.yaml``` and train with the CROHME dataset:

```
python train_stage.py --dataset CROHME
```

By default the ```batch size``` is set to 8 and you may need to use a GPU with 32GB RAM to train your model. 

If you want to use Adaptive Batch Sampler so that the batchsize can be altered to avoid OOM, run the code by:
```
python train_stage_ABS.py --dataset CROHME
```

