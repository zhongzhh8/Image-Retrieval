## Image Retrieval

### Introduction

This code implements an image retrieval task on two dataset: CUB200_2011 and Stanford Dogs. 

### Approach

I implement the approach in paper  [*Simultaneous Feature Learning and Hash Coding with Deep Neural Networks*](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Lai_Simultaneous_Feature_Learning_2015_CVPR_paper.html). And its network is proposed as fellow:

![](./Snipaste_2019-08-24_20-25-07.png)

As the paper was published early in 2015, the shared sub-network and divide-and-encode module in the network have been out of style. In order to enhance the effect of the network, I replace the sub-network with a pre-trained ResNet-18 as well as replace the divide-and-encode module with a fully connected yer.

#### what is hashing?

With the ever-growing large-scale image data on the Web, much attention has been devoted to nearest neighbor search via hashing methods. Learning-based hashing is an emerging stream of hash methods
that learn similarity-preserving hash functions to encode input data points (e.g., images) into binary codes. A good hashing function is able to reduce the required memory and  improve the retrieval speed by a large margin.

#### Triplet Loss

Triplet ranking loss is designed to characterize that one image is more similar to the second image than
to the third one.

![](./Snipaste_2019-08-24_20-55-30.png)

F(I), F(I+) ,F(I-) denote the embeddings of the query image, similar image and dissimilar image respectively.

#### Code Process 

1. Train the network with triplet loss on the training set for 1000~4000 epochs.
2. Input the training set and testing set into the network to get embeddings and then turn the embeddings into binary hash codes with a simple quantization function[^torch.sign()].
3. Use testing sample as querie to retrieve images from training samples. Calculate distance between binary codes of testing sample and training samples with Hamming distance. Use mAP to estimate the model's performance.



### Prerequisites

In order to run this code you will need to install:

1. Python3
2. Pytorch 0.4

### Usage

1. Firstly download and unzip the two datasets.
2. Change the arguments datapath_CUB and data_path_dog in main.py to indicate the file path.
3. Run the command bellow. 

```python
python main.py --dataset_name 'CUB' --binary_bits 64 --margin 20  --lr 0.001 --ngpu 1
```
And you can change the parameters if you want. I recommand to set margin as 20 when binary_bits is 64 and 8 when binary_bits is 16. It is suggested to set the value of margin as  a bit bigger than a quarter of the value of binary_bits.

### Result

Common parameters：

| lr    | batch size | optimizer | num_epochs |
| ----- | ---------- | --------- | ---------- |
| 0.001 | 80         | SGD       | 1000~4000  |

The result of mAP after 1000~4000 epochs training on the two datasets:

#### CUB

| binary bits | margin | baseline mAP | my mAP     |
| ----------- | :----- | ------------ | ---------- |
| 16          | 8      | 0.5137       | **0.6384** |
| 64          | 20     | 0.6949       | **0.7059** |

#### DOGS

| binary bits | margin | baseline mAP | my mAP     |
| ----------- | :----- | ------------ | ---------- |
| 16          | 8      | 0.6745       | **0.7127** |
| 64          | 20     | 0.7293       | **0.7615** |



### Reference

[1] Hanjiang Lai, Yan Pan, Ye Liu, Shuicheng Yan [*Simultaneous Feature Learning and Hash Coding with Deep Neural Networks*](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Lai_Simultaneous_Feature_Learning_2015_CVPR_paper.html)
[2] Hanjiang Lai, Jikai Chen, Libing Geng, Yan Pan, Xiaodan Liang, Jian Yin [*Improving Deep Binary Embedding Networks by Order-aware Reweighting of Triplets*](https://ieeexplore.ieee.org/abstract/document/8640819)