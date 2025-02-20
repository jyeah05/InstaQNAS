# InstaQNAS

![Image](https://github.com/user-attachments/assets/56b23748-5aec-49c1-8564-7bbef1224635)

- - -

## How to train policy & main networks

### 1. Pretrain
* Pretrain main network with randomly generated policies.
```shell
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
