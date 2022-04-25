import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from args import get_parser
from data.refer_data_module import fetch_data_loaders
from lavt import LAVT


def train(args):
    train_dl, val_dl, test_dl = fetch_data_loaders(args)

    wandb_logger = WandbLogger(project='lavt')
    if args.resume:
        print('Load the model from {}'.format(args.resume))
        model = LAVT.load_from_checkpoint(args.resume, args=args, num_train_steps=len(train_dl))
    else:
        model = LAVT(args=args, num_train_steps=len(train_dl))

    filename_fmt = '{}-{}-'.format(args.model_id, args.dataset) + '{epoch:02d}'
    checkpoint_callback = ModelCheckpoint(
        monitor='overall iou',
        dirpath='checkpoints/',
        filename=filename_fmt,
        save_top_k=3,
        mode='max')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=args.epochs,
        gpus=args.gpus,
        strategy='ddp',
        sync_batchnorm=True,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, lr_monitor_callback, ])
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


def test(args):
    # refer_data = ReferDataModule(args)
    _, _, test_dl = fetch_data_loaders(args)

    assert args.resume
    print('Load the model from {}'.format(args.resume))
    model = LAVT.load_from_checkpoint(args.resume, args=args, num_train_steps=0)

    trainer = pl.Trainer(
        gpus=args.gpus,
        num_sanity_val_steps=0)
    trainer.test(model=model, test_dataloaders=test_dl)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise ValueError('A mode should be either train or test: {}'.format(args.mode))
