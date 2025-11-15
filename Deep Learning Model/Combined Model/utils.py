import torch

class L1andAUCLoss(torch.nn.Module):
    '''一个是L1损失，一个是区分GFR是否大于40的AUC损失'''
    def __init__(self, auc_weight=0.5):
        super(L1andAUCLoss, self).__init__()
        self.auc_weight = auc_weight
        self.l1_loss = torch.nn.L1Loss()
        self.auc_loss = torch.nn.BCELoss()
    def forward(self, outputs, targets, use_zscore=False):
        if use_zscore:
            targets_40 = 0.6143095026760886
        else:
            targets_40 = 40.0
        l1 = self.l1_loss(outputs, targets)
        # 计算AUC损失
        targets_binary = (targets > targets_40).float()
        outputs_sigmoid = torch.sigmoid(outputs - targets_40)  # 将输出转换为概率
        auc_loss = self.auc_loss(outputs_sigmoid, targets_binary)
        total_loss = l1 + self.auc_weight * auc_loss
        # print(f"L1 Loss: {l1.item()}, AUC Loss: {auc_loss.item()}")
        return total_loss

def z_score_normalize(value):
    return (value - 27.952308474576274) / 19.61176161680865

def z_score_denormalize(z_value):
    return z_value * 19.61176161680865 + 27.952308474576274

def ccc_value(x, y):
    import numpy as np
    ''' Concordance Correlation Coefficient'''
    x = x.reshape(-1)
    y = y.reshape(-1)
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

def r(x, y):
    import numpy as np
    ''' Pearson Correlation Coefficient'''
    x = x.reshape(-1)
    y = y.reshape(-1)
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rho = sxy / (np.std(x)*np.std(y))
    return rho

def r2(x, y):
    ''' Coefficient of Determination (R-squared)'''
    from sklearn.metrics import r2_score
    x = x.reshape(-1)
    y = y.reshape(-1)
    return r2_score(x, y)

def draw_train(train_loss, val_loss, test_loss, ccc_train, ccc_val, ccc_test, rho_train, rho_val, rho_test):
    import matplotlib.pyplot as plt
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(18, 6))
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    # CCC plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, ccc_train, label='Train CCC')
    plt.plot(epochs, ccc_val, label='Val CCC')
    plt.plot(epochs, ccc_test, label='Test CCC')
    plt.xlabel('Epochs')
    plt.ylabel('CCC Value')
    plt.title('CCC per Epoch')
    plt.legend()
    # Pearson R plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, rho_train, label='Train Pearson R')
    plt.plot(epochs, rho_val, label='Val Pearson R')
    plt.plot(epochs, rho_test, label='Test Pearson R')
    plt.xlabel('Epochs')
    plt.ylabel('Pearson R Value')
    plt.title('Pearson R per Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model/all/training_metrics.png')
    plt.close()

def align_train_val(train_tensor, val_tensor, method="zscore", axis=(0, 2, 3), eps=1e-6):
        """对齐 Train 与 Validation 的特征分布。

        适用于形状类似于 (N, C, H, W) 或 (N, S, H, W) 的张量，其中第二维被视为“通道/切片”。

        参数:
            - train_tensor: torch.Tensor 或 numpy.ndarray，训练集张量。
            - val_tensor:   torch.Tensor 或 numpy.ndarray，验证集张量。
            - method:       "zscore" | "robust" | "meanstd_match"。
                    * zscore: 使用训练集的均值/标准差对 train/val 同步标准化（推荐部署时使用）。
                    * robust: 使用训练集的中位数/四分位距(IQR)做稳健缩放。
                    * meanstd_match: 先把各自做 zscore，再把验证集通过线性变换匹配到训练集的均值/方差
                        （等价于按通道做 μ/σ 匹配，常用于离线评估的域对齐）。
            - axis:        计算统计量时的归约维度，默认 (0, 2, 3) 表示对 N、H、W 归约，按通道保留。
            - eps:         数值稳定项。

        返回:
            (train_aligned, val_aligned, stats)
                - stats: { 'method': str, 'mean_train': 张量, 'std_train': 张量,
                                     'median_train': 张量, 'iqr_train': 张量,
                                     'mean_val': 张量, 'std_val': 张量 }

        说明:
            - 若输入是 numpy，会内部转成 torch 再计算，返回保持 torch 张量类型。
            - 通道维默认是 dim=1；如你的张量通道维不同，可自行调整 axis。
        """
        import torch
        import numpy as np
        to_torch = lambda x: x if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x))
        train = to_torch(train_tensor).float()
        val = to_torch(val_tensor).float()
        # 统一 axis 到正索引并排序去重
        ndim = train.dim()
        reduce_dims = tuple(sorted([(d if d >= 0 else ndim + d) for d in axis]))
        def _mean_std(x):
                mean = x.mean(dim=reduce_dims, keepdim=True)
                std = x.std(dim=reduce_dims, unbiased=False, keepdim=True).clamp_min(eps)
                return mean, std
        def _median_iqr(x):
                median = x.quantile(0.5, dim=reduce_dims, keepdim=True)
                q1 = x.quantile(0.25, dim=reduce_dims, keepdim=True)
                q3 = x.quantile(0.75, dim=reduce_dims, keepdim=True)
                iqr = (q3 - q1).clamp_min(eps)
                return median, iqr
        stats = { 'method': method }
        if method == "zscore":
                mu_tr, sd_tr = _mean_std(train)
                train_z = (train - mu_tr) / sd_tr
                val_z = (val - mu_tr) / sd_tr
                stats.update({ 'mean_train': mu_tr, 'std_train': sd_tr })
                return train_z, val_z, stats
        elif method == "robust":
                med_tr, iqr_tr = _median_iqr(train)
                train_r = (train - med_tr) / iqr_tr
                val_r = (val - med_tr) / iqr_tr
                stats.update({ 'median_train': med_tr, 'iqr_train': iqr_tr })
                return train_r, val_r, stats
        elif method == "meanstd_match":
                # 使用各自统计量标准化，然后把验证集线性拉回到训练集统计量
                mu_tr, sd_tr = _mean_std(train)
                mu_va, sd_va = _mean_std(val)
                train_n = (train - mu_tr) / sd_tr
                # 先标准化 val 到零均值单位方差，再缩放/平移到 train 的均值方差
                val_n = (val - mu_va) / sd_va
                val_matched = val_n * sd_tr + mu_tr
                stats.update({ 'mean_train': mu_tr, 'std_train': sd_tr, 'mean_val': mu_va, 'std_val': sd_va })
                return train_n, val_matched, stats
        else:
                raise ValueError(f"Unsupported method: {method}")

def draw_true_and_pred(y_true, y_pred, name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.savefig('model/all/true_vs_predicted_' + name + '.png')
    plt.close()

class CCC_Loss(torch.nn.Module):
    """Concordance Correlation Coefficient Loss"""
    def __init__(self):
        super(CCC_Loss, self).__init__()

    def forward(self, x, y):
        x = x.reshape(-1)
        y = y.reshape(-1)
        sxy = torch.sum((x - x.mean()) * (y - y.mean())) / x.shape[0]
        rhoc = 2 * sxy / (x.var() + y.var() + (x.mean() - y.mean())**2)
        return 1 - rhoc  # Return 1 - CCC for loss

def draw_roc(y_true, y_pred, filter, name):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    import torch
    import json
    with open('3D/parameter.json', 'r', encoding='utf-8') as file:
        parameter = json.load(file)
    model_id = parameter['model_id']

    y_true = torch.tensor(y_true).reshape(-1).cpu().numpy()
    y_pred = torch.tensor(y_pred).reshape(-1).cpu().numpy()
    y_true[y_true <= filter] = 0
    y_true[y_true > filter] = 1
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.4f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'model/all/roc_curve_' + name + '_' + str(filter) + '.png')
    plt.close()

def draw_confusion_matrix(y_true, y_pred, name):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import itertools
    import numpy as np
    import json
    with open('3D/parameter.json', 'r', encoding='utf-8') as file:
        parameter = json.load(file)
    model_id = parameter['model_id']

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    y_true[y_true <= 10] = 0
    y_true[(y_true > 10) & (y_true <= 40)] = 1
    y_true[y_true > 40] = 2
    y_pred[y_pred <= 10] = 0
    y_pred[(y_pred > 10) & (y_pred <= 40)] = 1
    y_pred[y_pred > 40] = 2
    # 绘制混淆矩阵热图
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ['GFR<=10', '10<GFR<=40', 'GFR>40'])
    plt.yticks(tick_marks, ['GFR<=10', '10<GFR<=40', 'GFR>40'])
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'./model/all/confusion_matrix_' + name + '.png')
    plt.close()

if __name__ == "__main__":
    import numpy as np
    # Example usage
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1.1, 2.1, 2.9, 4.2, 5.1]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    print("CCC:", ccc_value(y_true, y_pred))
    print("R value:", r(y_true, y_pred))