from UNet import mlp
from dataloader import create_dataloader
import torch
from tqdm import tqdm
from utils import ccc_value, r2, draw_train, draw_true_and_pred, CCC_Loss, draw_confusion_matrix, draw_roc
import numpy as np
import random
import pandas as pd

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = mlp(n_channels=1, n_classes=1)
model = model.to(device)

train_loader, val_loader = create_dataloader('nfyy', batch_size=8, shuffle=False, use_aligned=True, split=True)
test_loader = create_dataloader('gzyy', batch_size=8, shuffle=False, use_aligned=True, split=False)
gdph_loader = create_dataloader('gdph', batch_size=8, shuffle=False, use_aligned=True, split=False)

model.load_state_dict(torch.load(f'model/all/best_model.pth'))

model.eval()
with torch.no_grad():
    x_train = []
    y_train = []
    idx_train = []
    for batch in tqdm(train_loader):
        images_arterial, images_venous, left_gfr, right_gfr, idx, clin, pre_1, pre_2 = batch
        images_arterial = images_arterial.to(device).to(torch.float32)
        images_venous = images_venous.to(device).to(torch.float32)
        left_gfr = left_gfr.to(device).reshape(-1, 1).to(torch.float32)
        right_gfr = right_gfr.to(device).reshape(-1, 1).to(torch.float32)
        pre_1 = pre_1.to(device).reshape(-1, 1).to(torch.float32)
        pre_2 = pre_2.to(device).reshape(-1, 1).to(torch.float32)
        gfr = torch.cat([left_gfr, right_gfr], dim=1).reshape(-1, 1)
        clin = clin.to(device).to(torch.float32)
        outputs = model(images_arterial, images_venous, clin, pre_1, pre_2)
        outputs = outputs.reshape(-1, 1)
        x_train = x_train + gfr.cpu().numpy().tolist()
        y_train = y_train + outputs.cpu().detach().numpy().tolist()
        # name_train = name_train + name
        idx_train = idx_train + idx.reshape(-1, 1).cpu().numpy().tolist()
    x_train = np.array(x_train).reshape(-1)
    y_train = np.array(y_train).reshape(-1)
    # 找出y_train中大于30的值对应的x_train和y_train
    x_train_30 = x_train[y_train > 30]
    y_train_30 = y_train[y_train > 30]
    print(f'Train CCC: {ccc_value(x_train, y_train)}, R2: {r2(x_train, y_train)}')
    df = pd.DataFrame({'gfr': x_train, 'pred': y_train, 'diff': y_train - x_train, 'idx': np.array(idx_train).reshape(-1)})
    df.to_csv(f'model/all/train_results.csv', index=False)

    x_val = []
    y_val = []
    idx_val = []
    for batch in tqdm(val_loader):
        images_arterial, images_venous, left_gfr, right_gfr, idx, clin, pre_1, pre_2 = batch
        images_arterial = images_arterial.to(device).to(torch.float32)
        images_venous = images_venous.to(device).to(torch.float32)
        left_gfr = left_gfr.to(device).reshape(-1, 1).to(torch.float32)
        right_gfr = right_gfr.to(device).reshape(-1, 1).to(torch.float32)
        pre_1 = pre_1.to(device).reshape(-1, 1).to(torch.float32)
        pre_2 = pre_2.to(device).reshape(-1, 1).to(torch.float32)
        gfr = torch.cat([left_gfr, right_gfr], dim=1).reshape(-1, 1)
        clin = clin.to(device).to(torch.float32)
        outputs = model(images_arterial, images_venous, clin, pre_1, pre_2)
        outputs = outputs.reshape(-1, 1)
        x_val = x_val + gfr.cpu().numpy().tolist()
        y_val = y_val + outputs.cpu().detach().numpy().tolist()
        idx_val = idx_val + idx.reshape(-1, 1).cpu().numpy().tolist()
    x_val = np.array(x_val).reshape(-1)
    y_val = np.array(y_val).reshape(-1)
    x_val_30 = x_val[y_val > 30]
    y_val_30 = y_val[y_val > 30]
    print(f'Val CCC: {ccc_value(x_val, y_val)}, R2: {r2(x_val, y_val)}')
    df = pd.DataFrame({'gfr': x_val, 'pred': y_val, 'diff': y_val - x_val, 'idx': np.array(idx_val).reshape(-1)})
    df.to_csv(f'model/all/val_results.csv', index=False)

    x_test = []
    y_test = []
    idx_test = []
    for batch in tqdm(test_loader):
        images_arterial, images_venous, left_gfr, right_gfr, idx, clin, pre_1, pre_2 = batch
        images_arterial = images_arterial.to(device).to(torch.float32)
        images_venous = images_venous.to(device).to(torch.float32)
        left_gfr = left_gfr.to(device).reshape(-1, 1).to(torch.float32)
        right_gfr = right_gfr.to(device).reshape(-1, 1).to(torch.float32)
        pre_1 = pre_1.to(device).reshape(-1, 1).to(torch.float32)
        pre_2 = pre_2.to(device).reshape(-1, 1).to(torch.float32)
        gfr = torch.cat([left_gfr, right_gfr], dim=1).reshape(-1, 1)
        clin = clin.to(device).to(torch.float32)
        outputs = model(images_arterial, images_venous, clin, pre_1, pre_2)
        outputs = outputs.reshape(-1, 1)
        x_test = x_test + gfr.cpu().numpy().tolist()
        y_test = y_test + outputs.cpu().detach().numpy().tolist()
        idx_test = idx_test + idx.reshape(-1, 1).cpu().numpy().tolist()
    x_test = np.array(x_test).reshape(-1)
    y_test = np.array(y_test).reshape(-1)
    x_test_30 = x_test[y_test > 30]
    y_test_30 = y_test[y_test > 30]
    print(f'Test CCC: {ccc_value(x_test, y_test)}, R2: {r2(x_test, y_test)}')
    df = pd.DataFrame({'gfr': x_test, 'pred': y_test, 'diff': y_test - x_test, 'idx': np.array(idx_test).reshape(-1)})
    df.to_csv(f'model/all/test_results.csv', index=False)

    x_gdph = []
    y_gdph = []
    idx_gdph = []
    for batch in tqdm(gdph_loader):
        images_arterial, images_venous, left_gfr, right_gfr, idx, clin, pre_1, pre_2 = batch
        images_arterial = images_arterial.to(device).to(torch.float32)
        images_venous = images_venous.to(device).to(torch.float32)
        left_gfr = left_gfr.to(device).reshape(-1, 1).to(torch.float32)
        right_gfr = right_gfr.to(device).reshape(-1, 1).to(torch.float32)
        pre_1 = pre_1.to(device).reshape(-1, 1).to(torch.float32)
        pre_2 = pre_2.to(device).reshape(-1, 1).to(torch.float32)
        gfr = torch.cat([left_gfr, right_gfr], dim=1).reshape(-1, 1)
        clin = clin.to(device).to(torch.float32)
        outputs = model(images_arterial, images_venous, clin, pre_1, pre_2)
        outputs = outputs.reshape(-1, 1)
        x_gdph = x_gdph + gfr.cpu().numpy().tolist()
        y_gdph = y_gdph + outputs.cpu().detach().numpy().tolist()
        idx_gdph = idx_gdph + idx.reshape(-1, 1).cpu().numpy().tolist()
    x_gdph = np.array(x_gdph).reshape(-1)
    y_gdph = np.array(y_gdph).reshape(-1)
    print(f'GDPH CCC: {ccc_value(x_gdph, y_gdph)}, R2: {r2(x_gdph, y_gdph)}')
    df = pd.DataFrame({'gfr': x_gdph, 'pred': y_gdph, 'diff': y_gdph - x_gdph, 'idx': np.array(idx_gdph).reshape(-1)})
    df.to_csv(f'model/all/gdph_results.csv', index=False)

    draw_true_and_pred(x_val, y_val, f'Val')
    draw_true_and_pred(x_train, y_train, f'Train')
    draw_true_and_pred(x_test, y_test, f'Test')
    draw_true_and_pred(x_gdph, y_gdph, f'GDPH')
    draw_confusion_matrix(x_val, y_val, f'Val')
    draw_confusion_matrix(x_train, y_train, f'Train')
    draw_confusion_matrix(x_test, y_test, f'Test')
    draw_confusion_matrix(x_gdph, y_gdph, f'GDPH')
    draw_roc(x_val, y_val, 10, f'Val')
    draw_roc(x_train, y_train, 10, f'Train')
    draw_roc(x_test, y_test, 10, f'Test')
    draw_roc(x_gdph, y_gdph, 10, f'GDPH')
    draw_roc(x_val, y_val, 40, f'Val')
    draw_roc(x_train, y_train, 40, f'Train')
    draw_roc(x_test, y_test, 40, f'Test')
    draw_roc(x_gdph, y_gdph, 40, f'GDPH')
