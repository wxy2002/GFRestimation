'''def L1_loss(pred, target):
    import torch.nn.functional as F
    return F.l1_loss(pred, target)

def L1_loss_diff_importance(pred, target, alpha=0.5, threshold=60):
    # 不同大小traget值的重要性不同
    import torch.nn.functional as F
    import torch
    loss = torch.abs(pred - target)
    loss_large = loss[target >= threshold]
    loss_small = loss[target < threshold]
    return alpha * torch.mean(loss_large) + (1 - alpha) * torch.mean(loss_small)'''

class L1_loss_diff_importance:
    def __init__(self, alpha=0.5, threshold=60):
        self.alpha = alpha
        self.threshold = threshold

    def __call__(self, pred, target):
        import torch.nn.functional as F
        # 获取traget中大于等于threshold和小于threshold的索引
        index_large = target >= self.threshold
        index_small = target < self.threshold
        loss = self.alpha * F.l1_loss(pred[index_large], target[index_large]) + (1 - self.alpha) * F.l1_loss(pred[index_small], target[index_small])
        return loss

class L1_loss:
    def __init__(self):
        pass

    def __call__(self, pred, target):
        import torch.nn.functional as F
        return F.l1_loss(pred, target)
