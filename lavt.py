import operator
from argparse import Namespace
from functools import reduce
from itertools import chain
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

from bert.modeling_bert import BertModel
from lib import segmentation

EVAL_IOU_LIST = [.5, .6, .7, .8, .9]


def criterion(prediction: Tensor, target: Tensor) -> Tensor:
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return cross_entropy(prediction, target, weight=weight)


def items_from_batch(batch) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    image, target, sentences, attentions = batch
    sentences = sentences.squeeze(1)
    attentions = attentions.squeeze(1)
    return image, target, sentences, attentions


class LAVT(pl.LightningModule):
    def __init__(self, args: Namespace, num_train_steps: int):
        pl.LightningModule.__init__(self)
        self.args = args
        self.num_train_steps = num_train_steps

        self.model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
        self.bert_model = BertModel.from_pretrained(args.ck_bert)
        self.params_to_optimize = []
        self.register_parameter_to_optimize()

    def register_parameter_to_optimize(self):
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.params_to_optimize,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            amsgrad=self.args.amsgrad)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / (self.num_train_steps * self.args.epochs)) ** 0.9)
        return [optimizer], [lr_scheduler]

    def forward(self, batch, batch_idx):
        image, target, sentences, attentions = items_from_batch(batch)

        last_hidden_states = self.bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
        embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
        output = self.model(image, embedding, l_mask=attentions)
        return output, target

    def training_step(self, batch, batch_idx):
        output, target = self.forward(batch, batch_idx)

        loss = criterion(output, target)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output, target = self.forward(batch, batch_idx)

        prediction = output.argmax(1)
        intersection = torch.sum(torch.mul(prediction, target))
        union = torch.sum(torch.add(prediction, target)) - intersection
        self.log('val/loss', criterion(output, target), on_step=False, on_epoch=True)
        return {'i': intersection, 'u': union}

    def validation_step_end(self, batch_parts):
        intersection = batch_parts['i']
        union = batch_parts['u']
        if intersection == 0 or union == 0:
            iou = 0.
        else:
            iou = float(intersection) / float(union)
        return {'i': intersection, 'u': union, 'iou': iou}

    def test_step(self, batch, batch_idx):
        image, target, sentences, attentions = items_from_batch(batch)

        target = target.cpu().data.numpy()
        res = []
        for j in range(sentences.size(-1)):
            last_hidden_states = self.bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
            embedding = last_hidden_states.permute(0, 2, 1)
            output = self.model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
            output = output.cpu()
            output_mask = output.argmax(1).data.numpy()
            i = np.sum(np.logical_and(output_mask, target))
            u = np.sum(np.logical_or(output_mask, target))
            if u == 0:
                iou = 0.0
            else:
                iou = i * 1.0 / u
            res.append({'i': i, 'u': u, 'iou': iou})
        return res

    def test_epoch_end(self, test_step_outputs) -> None:
        self.accumulate_outputs(test_step_outputs)

    def validation_epoch_end(self, validation_step_outputs):
        self.accumulate_outputs(validation_step_outputs)

    def test_epoch_end(self, test_step_outputs):
        self.accumulate_outputs(list(chain.from_iterable(test_step_outputs)))

    def accumulate_outputs(self, outputs):
        num_iter = 0
        ci, cu = 0, 0
        acc_iou = 0.
        mean_iou = []
        seg_correct = [0 for _ in range(len(EVAL_IOU_LIST))]

        for output in outputs:
            num_iter += 1
            ci += output['i']
            cu += output['u']
            acc_iou += output['iou']
            mean_iou.append(output['iou'])

            for i, eval_iou in enumerate(EVAL_IOU_LIST):
                seg_correct[i] += (output['iou'] >= eval_iou)

            self.log('mean iou', np.mean(np.array(mean_iou)) * 100., on_step=False, on_epoch=True)
            for i, eval_iou in enumerate(EVAL_IOU_LIST):
                self.log('precision@{:.2f}'.format(eval_iou), seg_correct[i] * 100. / num_iter,
                         on_step=False, on_epoch=True)
            self.log('overall iou', ci * 100. / cu, on_step=False, on_epoch=True)
