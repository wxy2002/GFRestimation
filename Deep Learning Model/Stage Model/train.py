# from UNet import UNet3D
from dataloader import create_dataloader
import torch
from tqdm import tqdm
from utils import ccc_value, r2, draw_train, draw_true_and_pred, CCC_Loss, draw_confusion_matrix, draw_roc
import numpy as np
import random
import pandas as pd
import os
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
if os.path.exists(f'./model/{model_id}') is False:
    os.makedirs(f'./model/{model_id}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet3D(n_channels=1, n_classes=1)
model = model.to(device)

train_loader, val_loader = create_dataloader('nfyy', batch_size=8, shuffle=True, use_aligned=False, split=True, stage=stage_name, side=side)
print(len(train_loader))
# test_loader = create_dataloader('gzyy', batch_size=8, shuffle=False, use_aligned=False)

# criterion = CCC_Loss()
# criterion = torch.nn.HuberLoss()
criterion = torch.nn.L1Loss()
# criterion = torch.nn.MSELoss()
# criterion = torch.nn.SmoothL1Loss()
from utils import L1andAUCLoss
# criterion = L1andAUCLoss(auc_weight=0.5)
print(f'Using {criterion}')
learning_rate = 1e-4
no_weight_decay_params = []
weight_decay_params = []
for name, param in model.named_parameters():
    if 'bias' in name or 'bn' in name:
        no_weight_decay_params.append(param)
    else:
        weight_decay_params.append(param)
optimizer = torch.optim.AdamW([
    {'params': no_weight_decay_params, 'weight_decay': 0.0},
    {'params': weight_decay_params, 'weight_decay': 0.01}
], lr=learning_rate, weight_decay=0.01)

total_epochs = 200
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=total_epochs // 10)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - total_epochs // 10, eta_min=learning_rate * 0.1)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine_scheduler], milestones=[total_epochs // 10])

train_loss_all = []
ccc_train_all = []
val_loss_all = []
ccc_val_all = []
test_loss_all = []
ccc_test_all = []
rho_train_all = []
rho_val_all = []
rho_test_all = []
max_ccc = -2.0
max_ccc_epoch = -1
for epoch in range(total_epochs):
    model.train()
    train_loss = 0.0
    train_loss1 = 0.0
    train_loss2 = 0.0
    print(f"Epoch {epoch + 1} started.")
    x_train = []
    y_train = []
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        images, left_gfr, right_gfr, clin = batch
        images = images.to(device).to(torch.float32)
        left_gfr = left_gfr.to(device).reshape(-1, 1).to(torch.float32)
        right_gfr = right_gfr.to(device).reshape(-1, 1).to(torch.float32)
        gfr = torch.cat([left_gfr, right_gfr], dim=1).reshape(-1, 1)
        clin = clin.to(device).reshape(-1, 2).to(torch.float32)
        outputs = model(images, clin)
        outputs = outputs.reshape(-1, 1)
        optimizer.zero_grad()
        loss = criterion(outputs, gfr)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        x_train = x_train + gfr.cpu().numpy().tolist()
        y_train = y_train + outputs.cpu().detach().numpy().tolist()
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    ccc_train = ccc_value(x_train, y_train)
    rho_train = r2(x_train, y_train)
    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, CCC: {ccc_train}, Pearson R: {rho_train}")
    scheduler.step()
    
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        ccc_val = 0.0
        x_val = []
        y_val = []
        for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}"):
            images, left_gfr, right_gfr, clin = batch
            images = images.to(device).to(torch.float32)
            left_gfr = left_gfr.to(device).reshape(-1, 1).to(torch.float32)
            right_gfr = right_gfr.to(device).reshape(-1, 1).to(torch.float32)
            gfr = torch.cat([left_gfr, right_gfr], dim=1).reshape(-1, 1)
            clin = clin.to(device).reshape(-1, 2).to(torch.float32)
            outputs = model(images, clin)
            outputs = outputs.reshape(-1, 1)
            x_val = x_val + gfr.cpu().numpy().tolist()
            y_val = y_val + outputs.cpu().detach().numpy().tolist()
            val_loss += criterion(outputs, gfr).item()
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        val_loss /= len(val_loader)
        ccc_val = ccc_value(x_val, y_val)
        rho_val = r2(x_val, y_val)

        ccc_test = 0.0
        rho_test = 0.0
        test_loss = 0.0
        x = [0]
        y = [0]
        '''for batch in tqdm(test_loader, desc=f"Testing Epoch {epoch + 1}"):
            images, left_gfr, right_gfr = batch
            images = images.to(device).to(torch.float32)
            left_gfr = left_gfr.to(device).reshape(-1, 1).to(torch.float32)
            right_gfr = right_gfr.to(device).reshape(-1, 1).to(torch.float32)
            gfr = torch.cat([left_gfr, right_gfr], dim=1).reshape(-1, 1)
            # gfr = left_gfr.reshape(-1, 1)  # 只使用左侧GFR进行测试
            outputs = model(images).reshape(-1, 1)
            x = x + gfr.cpu().numpy().tolist()
            y = y + outputs.cpu().detach().numpy().tolist()
            test_loss += criterion(outputs, gfr).item()
        test_loss /= len(test_loader)
        x = np.array(x)
        y = np.array(y)
        ccc_test = ccc_value(x, y)
        rho_test = r2(x, y)'''
    print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, CCC: {ccc_val}, R Square: {rho_val}")
    print(f"Epoch {epoch + 1}, Test Loss: {test_loss}, CCC: {ccc_test}, R Square: {rho_test}")
    print(f"Epoch {epoch + 1} completed.\n")
    train_loss_all.append(train_loss)
    ccc_train_all.append(ccc_train)
    rho_train_all.append(rho_train)
    val_loss_all.append(val_loss)
    ccc_val_all.append(ccc_val)
    rho_val_all.append(rho_val)
    test_loss_all.append(test_loss)
    ccc_test_all.append(ccc_test)
    rho_test_all.append(rho_test)
    draw_train(train_loss_all, val_loss_all, test_loss_all, 
               ccc_train_all, ccc_val_all, ccc_test_all, 
               rho_train_all, rho_val_all, rho_test_all)
    if ccc_val > max_ccc:
        max_ccc = ccc_val
        max_ccc_epoch = epoch + 1
        torch.save(model.state_dict(), f'./model/{model_id}/best_model_{stage_name}.pth')
        print(f"New best model saved with CCC: {max_ccc} at epoch {max_ccc_epoch}\n")
        draw_true_and_pred(x_val, y_val, f'Val_{stage_name}')
        draw_true_and_pred(x_train, y_train, f'Train_{stage_name}')
        draw_confusion_matrix(x_val, y_val, f'Val_{stage_name}')
        draw_confusion_matrix(x_train, y_train, f'Train_{stage_name}')
    else:
        print(f"No improvement in CCC: {ccc_val}. Best CCC Train {ccc_train_all[max_ccc_epoch-1]}, val {ccc_val_all[max_ccc_epoch-1]}, test {ccc_test_all[max_ccc_epoch-1]}")
        print(f'Loss in train {train_loss_all[max_ccc_epoch-1]}, val {val_loss_all[max_ccc_epoch-1]}, test {test_loss_all[max_ccc_epoch-1]}\n')