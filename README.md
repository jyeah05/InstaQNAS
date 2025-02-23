# InstaQNAS

## Abstract
To reduce computational redundancy inherent in fixed-bit-width quantization, input-adaptive quantization dynamically adjusts the bit-width of network parameters based on the difficulty of the given input. However, estimating image difficulty for object detection is a non-trivial task, as multiple detection results may occur within a single image. In this paper, we propose an input-adaptive mixed-precision framework that automatically adjusts the bit-width of each layer in the target model based on the characteristics of an input image. For searching optimal bit configurations, the framework employs a reward function that considers both the difficulty of a single image and the computational cost. Experimental results demonstrate that the proposed method outperforms prior quantization methods with fixed bit-widths.

## Overall Process
![Image](https://github.com/user-attachments/assets/56b23748-5aec-49c1-8564-7bbef1224635)

## Reward Function for InstaQNAS
![Image](https://github.com/user-attachments/assets/f0c6a076-edc5-4fbe-9992-fa33603e78a0)
- - -

## How to train policy & main networks

### 0. Environment Setting
* You need to change the directories for dataset and checkpoint for VOC training.
    * L562, 575, 715 in `search_multinomial.py`
    * L843 in `pretrain_ssd.py`
    * L162, 165 in `dataloader.py`
    * checkpoint arguments in shell scriptsi
* Please prepare pytorch environment
* Please prepare VOC07+12 dataset

### 1. Pretrain(Action)
* Pretrain main network with randomly generated policies.
    * We used full-precision pretrain model of https://github.com/qfgaohao/pytorch-ssd
    * Set the `resume_path` argument in `pretarin.sh` to the directory of the downloaded pretrained model
```shell
sh pretrain_mbv1_fp.sh   
sh pretrain_mbv1.sh
``` 
### 2. Search(Exploration)
* Train policy network to search the optimal bit policies for the main network.
```shell
sh search_mbv1.sh
```
### 3. Finetune
* Finetune the main networks with searched policies.
```shell
sh finetune_mbv1.sh
```

### Citation
* If you find our repository useful, please cite our paper in your publications:
```shell
@article{jeong2025instaqnas,
  title={Input-Adaptive Mixed-Precision Framework for Efficient Object Detection},
  author={Jeong, Issac and Jeon, Ji-Ye and Kim, Kihwan and Jeon, Dongsuk and Lee, Hyuk-Jae},
  booktitle={IEEE International Symposium on Circuits and Systems (ISCAS)},
  year={2025}
}
```
