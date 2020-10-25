# Applying FixMatch to NAVER Shopping Dataset
Yoonseo Kim, Yeji Han

### Setup
You can install this repository using ```git clone``` 
```bash
git clone https://github.com/yejihan-dev/cs492-cv-project.git
```

You should log in to NSML to run the model.
```bash
nsml login
```

### Train the model
To train the model in NSML, use 
```bash
nsml run -d fashion_dataset -g 1 --args "--name resnet18 --optim sgd --batch-size 64 --epoch 200 --mu 7 --lambda-u 1 --threshold 0.95"
```

### Load Pretrained Model
We saved a pretrained model in path `/pretrained/model.pt`. You can load this pretrained weights by editing above command into

```bash
nsml run -d fashion_dataset -g 1 --args "--name resnet18 --optim sgd --batch-size 64 --epoch 200 --mu 7 --lambda-u 1 --threshold 0.95"
```


#### Multi-GPU training
Just pass more GPUs and fixmatch automatically scales to them, here we assign GPUs 4-7 to the program:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python fixmatch.py --filters=32 --dataset=cifar10.3@40-1 --train_dir ./experiments/fixmatch
```

#### Flags

```bash
python fixmatch.py --help
# The following option might be too slow to be really practical.
# python fixmatch.py --helpfull
# So instead I use this hack to find the flags:
fgrep -R flags.DEFINE libml fixmatch.py
```

The `--augment` flag can use a little more explanation. It is composed of 3 values, for example `d.d.d`
(`d`=default augmentation, for example shift/mirror, `x`=identity, e.g. no augmentation, `ra`=rand-augment,
 `rac`=rand-augment + cutout):
- the first `d` refers to data augmentation to apply to the labeled example. 
- the second `d` refers to data augmentation to apply to the weakly augmented unlabeled example. 
- the third `d` refers to data augmentation to apply to the strongly augmented unlabeled example. For the strong
augmentation, `d` is followed by `CTAugment` for `fixmatch.py` and code inside `cta/` folder.


## Checkpoint accuracy



## Reference
[1] "[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)" by Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel.

[2] "[MixMatch - A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)" by David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver and Colin Raffel.
