# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader

class Model(nn.Module):
    def __init__(self, encoder, config):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config

    def forward(self, input_ids=None,  labels=None):
        if input_ids is not None:
            logits = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(0))[0]
        # elif input_embeddings is not None:
        #     logits = self.encoder(input_embeds=input_embeddings, attention_mask=input_masks)[0]
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        self.eval()
        logits = []
        labels = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs, label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())


        probs = [[1 - prob[0], prob[0]] for prob in logits]
        pred_labels = [1 if label else 0 for label in logits[:, 0] > 0.5]

        return probs, pred_labels
