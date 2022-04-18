import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from args import get_parser
from lavt import LAVTPL
from utils import get_dataset


def main(args):
    train_dataset, num_classes = get_dataset(split='train', args=args, eval_mode=False)
    val_dataset, _ = get_dataset(split='val', args=args, eval_mode=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers)

    wandb_logger = WandbLogger(project='lavt')
    model = LAVTPL(args=args, num_train_steps=len(train_loader))

    filename_fmt = '{}-{}-'.format(args.model_id, args.dataset) + '{epoch:02d}'
    checkpoint_callback = ModelCheckpoint(
        monitor='overall iou',
        dirpath='checkpoints/',
        filename=filename_fmt,
        save_top_k=3,
        mode='max')

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=args.epochs,
        gpus=args.gpus,
        strategy='ddp',
        sync_batchnorm=True,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, ])
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
