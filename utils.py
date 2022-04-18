import transforms as T


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    return T.Compose(transforms)


def get_dataset(split: str, args, eval_mode: bool):
    from data.dataset_refer_bert import ReferDataset
    transform = get_transform(args)
    ds = ReferDataset(args,
                      split=split,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=eval_mode)
    num_classes = 2
    return ds, num_classes
