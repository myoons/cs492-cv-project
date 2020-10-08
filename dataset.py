from torchvision.transforms import transforms

def weak_aug(resize, imsize):
    weak_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomResizedCrop(imsize),
        weak_augmentation.flip_augmentation(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return weak_transform


def strong_aug(resize, imsize):
    transform_train = transforms.Compose([
        transforms.Resize(opts.imResize),
        transforms.RandomResizedCrop(opts.imsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Add RandAugment with N, M(hyperparameter)
    return transform_train.transforms.insert(0, rand_augment(N, M))


def get_dataset():
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        './data', args.num_labeled, args.k_img, args.k_img * args.mu)

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    return labeled_trainloader, unlabeled_trainloader, test_loader