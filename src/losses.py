__author__ = 'Muhammed Patel'
__contributor__ = 'Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park'
__copyright__ = ['university of waterloo']
__contact__ = ['m32patel@uwaterloo.ca', 'xinweic@uwaterloo.ca']
__version__ = '1.0.0'
__date__ = '2024-04-05'

import torch
from torch import nn
import torch.nn.functional as F


class OrderedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(OrderedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        loss = criterion(output, target)
        # calculate the hard predictions by using softmax followed by an argmax
        softmax = torch.nn.functional.softmax(output, dim=1)
        hard_prediction = torch.argmax(softmax, dim=1)
        # set the mask according to ignore index
        mask = target == self.ignore_index
        hard_prediction = hard_prediction[~mask]
        target = target[~mask]
        # calculate the absolute difference between target and prediction
        weights = torch.abs(hard_prediction-target) + 1
        # remove ignored index losses
        loss = loss[~mask]
        # if done normalization with weights the loss becomes of the order 1e-5
        # loss = (loss * weights)/weights.sum()
        loss = (loss * weights)
        loss = loss.mean()

        return loss


class MSELossFromLogits(nn.Module):
    def __init__(self, chart, ignore_index=-100):
        super(MSELossFromLogits, self).__init__()
        self.ignore_index = ignore_index
        self.chart = chart
        if self.chart == 'SIC':
            self.replace_value = 11
            self.num_classes = 12
        elif self.chart == 'SOD':
            self.replace_value = 6
            self.num_classes = 7
        elif self.chart == 'FLOE':
            self.replace_value = 7
            self.num_classes = 8
        else:
            raise NameError('The chart \'{self.chart} \'is not recognized')

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        # replace ignore index value(for e.g 255) with a number 11. Becuz one hot encode requires
        # continous numbers (you cant one hot encode 255)
        target = torch.where(target == self.ignore_index,
                             torch.tensor(self.replace_value, dtype=target.dtype,
                                          device=target.device), target)
        # do one hot encoding
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2)

        # apply softmax on logits
        softmax = torch.softmax(output, dim=1, dtype=output.dtype)

        criterion = torch.nn.MSELoss(reduction='none')

        # calculate loss between softmax and one hot encoded target
        loss = criterion(softmax, target_one_hot.to(softmax.dtype))

        # drop the last channel since it belongs to ignore index value and should not
        # contribute to the loss

        loss = loss[:, :-1, :, :]
        loss = loss.mean()
        return loss

class WaterConsistencyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.keys = ['SIC', 'SOD', 'FLOE']
        self.activation = nn.Softmax(dim=1)
    
    def forward(self, output):
        sic = self.activation(output[self.keys[0]])[:, 0, :, :]
        sod = self.activation(output[self.keys[1]])[:, 0, :, :]
        floe = self.activation(output[self.keys[2]])[:, 0, :, :]
        return torch.mean((sic-sod)**2 + (sod-floe)**2 + (floe-sic)**2)

# only applicable to regression outputs
class MSELossWithIgnoreIndex(nn.MSELoss):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(MSELossWithIgnoreIndex, self).__init__(reduction=reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = (target != self.ignore_index).type_as(input)
        diff = input.squeeze(-1) - target
        diff = diff * mask
        loss = torch.sum(diff ** 2) / mask.sum()
        return loss

# only applicable to regression outputs
class MSELossWithIgnoreIndex(nn.MSELoss):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(MSELossWithIgnoreIndex, self).__init__(reduction=reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = (target != self.ignore_index).type_as(input)
        diff = input.squeeze(-1) - target
        diff = diff * mask
        loss = torch.sum(diff ** 2) / mask.sum()
        return loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, logits, target):
        """
        logits: [B, C, H, W]
        target: [B, H, W]
        """
        logp = self.ce(logits, target)  # [B,H,W]
        p = torch.exp(-logp)

        loss = ((1 - p) ** self.gamma) * logp

        mask = (target != self.ignore_index).float()
        return (loss * mask).sum() / (mask.sum() + 1e-6)
    
class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):

        num_classes = logits.shape[1]

        # =====================================================
        # valid mask FIRST (critical fix)
        # =====================================================
        valid_mask = (target != self.ignore_index)

        # replace ignore pixels with 0 (safe dummy class)
        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        probs = torch.softmax(logits, dim=1)

        target_onehot = torch.nn.functional.one_hot(
            target_safe,
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1).float()

        probs = probs * valid_mask
        target_onehot = target_onehot * valid_mask

        intersection = (probs * target_onehot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()
    
class FLOELoss(nn.Module):
    def __init__(self, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma)
        self.dice = DiceLoss()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, target):
        focal_loss = self.focal(logits, target)
        dice_loss = self.dice(logits, target)

        return (
            self.focal_weight * focal_loss +
            self.dice_weight * dice_loss
        )
