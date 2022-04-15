import gc
import operator
from functools import reduce

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

import transforms as T
import utils
from bert.modeling_bert import BertModel
from lib import segmentation


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    return T.Compose(transforms)


def get_dataset(split: str, args):
    from data.dataset_refer_bert import ReferDataset
    transform = get_transform(args)
    ds = ReferDataset(args,
                      split=split,
                      image_transforms=transform,
                      target_transforms=None)
    num_classes = 2
    return ds, num_classes


# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union


def criterion(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(input, target, weight=weight)


def evaluate(model, data_loader, bert_model):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
            output = model(image, embedding, l_mask=attentions)
            iou, I, U = IoU(output, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * iou, 100 * cum_I / cum_U


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions = data
        image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
        embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
        output = model(image, embedding, l_mask=attentions)

        loss = criterion(output, target)
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data, last_hidden_states, embedding
        gc.collect()
        torch.cuda.empty_cache()


class LAVTPL(pl.LightningModule):
    def __init__(self, args, num_train_steps):
        pl.LightningModule.__init__(self)
        self.args = args
        print(args.model)
        self.num_train_steps = num_train_steps

        self.model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
        self.bert_model = BertModel.from_pretrained(args.ck_bert)

        # if args.resume:
        #     checkpoint = torch.load(args.resume, map_location='cpu')
        #     self.model.load_state_dict(checkpoint['model'])
        #     self.bert_model.load_state_dict(checkpoint['bert_model'])

        backbone_no_decay = list()
        backbone_decay = list()
        for name, m in self.model.backbone.named_parameters():
            if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
                backbone_no_decay.append(m)
            else:
                backbone_decay.append(m)

        self.params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in self.model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in self.bert_model.encoder.layer[i].parameters() if p.requires_grad]
                               for i in range(10)])}]

        # start_time = time.time()
        # iterations = 0
        # best_oIoU = -0.1

        # if args.resume:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #     resume_epoch = checkpoint['epoch']
        # else:
        #     resume_epoch = -999

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.params_to_optimize,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            amsgrad=self.args.amsgrad)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / (self.num_train_steps * args.epochs)) ** 0.9)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        image, target, sentences, attentions = batch
        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        last_hidden_states = self.bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
        embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
        output = self.model(image, embedding, l_mask=attentions)

        return criterion(output, target)

    def validation_step(self, batch, batch_idx):
        image, target, sentences, attentions = batch

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        last_hidden_states = self.bert_model(sentences, attention_mask=attentions)[0]
        embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
        output = self.model(image, embedding, l_mask=attentions)
        iou, I, U = IoU(output, target)
        self.log('i', I)
        self.log('u', U)
        self.log('iou', iou)
        return criterion(output, target)


def main(args):
    train_dataset, num_classes = get_dataset('train', args=args)
    val_dataset, _ = get_dataset('val', args=args)

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

    model = LAVTPL(args=args, num_train_steps=len(train_loader))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        strategy='ddp',
        sync_batchnorm=True)
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    # utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
