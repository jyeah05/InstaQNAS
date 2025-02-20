# InstaQNAS

## Abstract
To reduce computational redundancy inherent in fixed-bit-width quantization, input-adaptive quantization dynamically adjusts the bit-width of network parameters based on the difficulty of the given input. However, estimating image difficulty for object detection is a non-trivial task, as multiple detection results may occur within a single image. In this paper, we propose an input-adaptive mixed-precision framework that automatically adjusts the bit-width of each layer in the target model based on the characteristics of an input image. For searching optimal bit configurations, the framework employs a reward function that considers both the difficulty of a single image and the computational cost. Experimental results demonstrate that the proposed method outperforms prior quantization methods with fixed bit-widths.

## Overall Process
![Image](https://github.com/user-attachments/assets/56b23748-5aec-49c1-8564-7bbef1224635)

## Reward Function for NAS
![Image](https://github.com/user-attachments/assets/f0c6a076-edc5-4fbe-9992-fa33603e78a0)
- - -

## How to train policy & main networks

### 0. Environment Setting


### 1. Pretrain
* Pretrain main network with randomly generated policies.
    * We used full-precision pretrain model of https://github.com/qfgaohao/pytorch-ssd
    * Set the `resume_path` argument in `pretarin.sh` to the directory of the downloaded pretrained model
```shell
sh pretrain_mbv1_fp.sh   
sh pretrain_mbv1.sh
``` 
### 2. Search
* Train policy network to search the optimized main network bit policies.
```shell
sh search_mbv1.sh
```
### 3. Finetune
* Finetune the main networks with searched policies.
```shell
sh finetuen_mbv1.sh
```
