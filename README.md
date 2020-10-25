# CS492 CV Project
## Applying FixMatch to NAVER Shopping Dataset

we introduce the variation of FixMatch applying to NAVER Shopping Dataset [2]. In this repository, we especially focus on optimizing augmentation step inside FixMatch [3]. We compare how the performance of FixMatch is better than that of MixMatch [1]. Next, we make an ablation study on main hyperparameters of our model to find obstacles and possible approaches for a better performance.


Project Contributor: [Yoonseo Kim](https://github.com/myoons), [Yeji Han](https://github.com/yejihan-dev)


## Setup
You can install this repository using ```git clone``` 
```bash
git clone https://github.com/yejihan-dev/cs492-cv-project.git
```

You should log in to NSML to run the model.
```bash
nsml login
```

## Code Architecture
`main.py` is what NSML runs to import the model or to train it. `main.py` serves as a frame for iterating epochs. In order to see the specific implementation of FixMatch alogrithm, refer to `fixmatch.py`. If you want to see how RandAugment is implemented, refer to `rand_augment.py`. To find out auxiliary functions for dataset and NSML, refer to `dataset.py` and `utils.py`.


## Train the model
To train the model on NSML, use the
```bash
nsml run -d fashion_dataset -g 1 --args "--name resnet18 --optim sgd --batch-size 64 --epoch 200 --mu 7 --lambda-u 1 --threshold 0.95"
```

### List of Major Flags
There are several option flags you can apply to change hyperparameters of the model.
- --name: image classifying architecture
    - default='resnet34'
    - type=str
    - choices=['resnet18', 'resnet34', 'resnet50']
- --optim
    - default='sgd'
    - type=str
    - choices=['adam', 'sgd', 'adamw', 'yogi']
- --batch-size, 
    - default=64, 
    - type=int
- --epochs
    - default=200
    - type=int
- --lr
    - default=0.03
    - type=float
- --mu: coefficient of unlabeled batch size
    - default=7
    - type=int
- --lambda-u: coefficient of unlabeled loss
    - default=1
    - type=float
- --threshold: pseudo-label threshold
    - default=0.95
    - type=float
- --mode: nsml mode
    - --mode
    - type=str
    - default='train'

## Load Pretrained Model
You can load this pretrained weights using `pretrained` flag in NSML. We also saved the best pretrained model in path `/pretrained/model.pt`.

```bash
nsml run -d fashion_dataset -g 1 --args "--name res34 --pretrained True"
```

## Ablation Study
If GPUs are sufficient on NSML, you can try automated ablation study by running `ablation_sh`. Since it automatically runs several session, GPUs may be congested.

```bash
./ablation_study.sh
```

## Reference
[1] "[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)" by Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel.

[2] "[MixMatch - A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)" by David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver and Colin Raffel.

[3] "[Randaugment: Practical automated data augmentation with a reduced search space.](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.pdf)" by Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V.