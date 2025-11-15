from dataloader import create_dataloader
import torch
from tqdm import tqdm
from utils import ccc_value, r2, draw_train, draw_true_and_pred, CCC_Loss, draw_confusion_matrix, draw_roc
import numpy as np
import random
import pandas as pd
import importlib
import json

with open('3D/parameter.json', 'r', encoding='utf-8') as file:
    parameter = json.load(file)
model_id = parameter['model_id']
path = f'模型备份.UNet_{model_id}'
module_to_import = importlib.import_module(path)
UNet3D = getattr(module_to_import, "UNet3D")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(42)

print(f'Parameter:')
print(parameter)
stage_name = parameter['stage_name']  # 'arterial', 'venous', 'delay'
side = parameter['side']
model_id = parameter['model_id']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet3D(n_channels=1, n_classes=1)
model = model.to(device)

train_loader, val_loader = create_dataloader('nfyy', batch_size=8, shuffle=True, use_aligned=False, 
                                             split=True, stage=stage_name, side=side, give_id=True)
test_loader = create_dataloader('gzyy', batch_size=8, shuffle=False, use_aligned=False, side='All', split=False, stage=stage_name)

model.load_state_dict(torch.load(f'model/{model_id}/best_model_{stage_name}.pth'))

model.eval()
with torch.no_grad():
    x_train = []
    y_train = []
    idx_train = []
    for batch in tqdm(train_loader):
        images, left_gfr, right_gfr, idx, clin = batch
        images = images.to(device).to(torch.float32)
        left_gfr = left_gfr.to(device).reshape(-1, 1).to(torch.float32)
        right_gfr = right_gfr.to(device).reshape(-1, 1).to(torch.float32)
        gfr = torch.cat([left_gfr, right_gfr], dim=1).reshape(-1, 1)
        clin = clin.to(device).reshape(-1, 2).to(torch.float32)
        outputs = model(images, clin)
        outputs = outputs.reshape(-1, 1)
        x_train = x_train + gfr.cpu().numpy().tolist()
        y_train = y_train + outputs.cpu().detach().numpy().tolist()
        # name_train = name_train + name
        idx_train = idx_train + idx.reshape(-1, 1).cpu().numpy().tolist()
    x_train = np.array(x_train).reshape(-1)
    y_train = np.array(y_train).reshape(-1)
    print(f'Train CCC: {ccc_value(x_train, y_train)}, R2: {r2(x_train, y_train)}')
    df = pd.DataFrame({'gfr': x_train, 'pred': y_train, 'diff': y_train - x_train, 'idx': np.array(idx_train).reshape(-1)})
    df.to_csv(f'model/{model_id}/{stage_name}_train_results.csv', index=False)

    x_val = []
    y_val = []
    idx_val = []
    for batch in tqdm(val_loader):
        images, left_gfr, right_gfr, idx, clin = batch
        images = images.to(device).to(torch.float32)
        left_gfr = left_gfr.to(device).reshape(-1, 1).to(torch.float32)
        right_gfr = right_gfr.to(device).reshape(-1, 1).to(torch.float32)
        gfr = torch.cat([left_gfr, right_gfr], dim=1).reshape(-1, 1)
        clin = clin.to(device).reshape(-1, 2).to(torch.float32)
        outputs = model(images, clin)
        outputs = outputs.reshape(-1, 1)
        x_val = x_val + gfr.cpu().numpy().tolist()
        y_val = y_val + outputs.cpu().detach().numpy().tolist()
        idx_val = idx_val + idx.reshape(-1, 1).cpu().numpy().tolist()
    x_val = np.array(x_val).reshape(-1)
    y_val = np.array(y_val).reshape(-1)
    print(f'Val CCC: {ccc_value(x_val, y_val)}, R2: {r2(x_val, y_val)}')
    df = pd.DataFrame({'gfr': x_val, 'pred': y_val, 'diff': y_val - x_val, 'idx': np.array(idx_val).reshape(-1)})
    df.to_csv(f'model/{model_id}/{stage_name}_val_results.csv', index=False)

    x_test = []
    y_test = []
    idx_test = []
    for batch in tqdm(test_loader):
        images, left_gfr, right_gfr, clin = batch
        images = images.to(device).to(torch.float32)
        left_gfr = left_gfr.to(device).reshape(-1, 1).to(torch.float32)
        right_gfr = right_gfr.to(device).reshape(-1, 1).to(torch.float32)
        gfr = torch.cat([left_gfr, right_gfr], dim=1).reshape(-1, 1)
        clin = clin.to(device).reshape(-1, 2).to(torch.float32)
        outputs = model(images, clin)
        outputs = outputs.reshape(-1, 1)
        x_test = x_test + gfr.cpu().numpy().tolist()
        y_test = y_test + outputs.cpu().detach().numpy().tolist()
    x_test = np.array(x_test).reshape(-1)
    y_test = np.array(y_test).reshape(-1)
    print(f'Test CCC: {ccc_value(x_test, y_test)}, R2: {r2(x_test, y_test)}')
    df = pd.DataFrame({'gfr': x_test, 'pred': y_test, 'diff': y_test - x_test})
    df.to_csv(f'model/{model_id}/{stage_name}_test_results.csv', index=False)

    draw_true_and_pred(x_val, y_val, f'Val_{stage_name}')
    draw_true_and_pred(x_train, y_train, f'Train_{stage_name}')
    draw_true_and_pred(x_test, y_test, f'Test_{stage_name}')
    draw_confusion_matrix(x_val, y_val, f'Val_{stage_name}')
    draw_confusion_matrix(x_train, y_train, f'Train_{stage_name}')
    draw_confusion_matrix(x_test, y_test, f'Test_{stage_name}')
    draw_roc(x_val, y_val, 10, f'Val_{stage_name}')
    draw_roc(x_train, y_train, 10, f'Train_{stage_name}')
    draw_roc(x_test, y_test, 10, f'Test_{stage_name}')
    draw_roc(x_val, y_val, 40, f'Val_{stage_name}')
    draw_roc(x_train, y_train, 40, f'Train_{stage_name}')
    draw_roc(x_test, y_test, 40, f'Test_{stage_name}')
