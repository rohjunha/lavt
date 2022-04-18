import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from args import get_parser
from data.refer_data_module import ReferDataModule
from lavt import LAVT


def train(args):
    refer_data = ReferDataModule(args)

    wandb_logger = WandbLogger(project='lavt')
    model = LAVT(args=args, num_train_steps=len(refer_data))

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
    trainer.fit(model=model, datamodule=refer_data)


def test(args):
    refer_data = ReferDataModule(args)

    assert args.resume
    print('Load the model from {}'.format(args.resume))
    model = LAVT.load_from_checkpoint(args.resume)

    trainer = pl.Trainer(
        gpus=args.gpus,
        strategy='ddp',
        num_sanity_val_steps=0)
    trainer.test(model=model, datamodule=refer_data)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise ValueError('A mode should be either train or test: {}'.format(args.mode))
