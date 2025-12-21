import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from data import get_data
import json

def sse_print(event: str, data: dict) -> str:
    """
    SSE 打印
    :param event: 事件名称
    :param data: 事件数据（字典或能被 json 序列化的对象）
    :return: SSE 格式字符串
    """
    # 将数据转成 JSON 字符串
    json_str = json.dumps(data, ensure_ascii=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj)
    
    # 按 SSE 协议格式拼接
    message = f"event: {event}\n" \
              f"data: {json_str}\n\n"
    print(message, flush=True, end='')
    return message

def train(model, target_label, train_loader, param):
    sse_print("processing_label", {"label": target_label, "message": f"Processing label: {target_label}"})

    width, height = param["image_size"]
    trigger = torch.rand((3, width, height), requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    Epochs = param["Epochs"]
    lamda = param["lamda"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.005)
    model.to(device)
    model.eval()

    for epoch in range(Epochs):
        norm = 0.0
        # 使用 tqdm 进度条，但将更新信息通过 SSE 发送
        # progress_data = {'epoch': epoch + 1, 'total_epochs': Epochs}
        # pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch + 1:3d}')
        for i, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():
                # 防止trigger和norm越界
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
            
            # 每10步发送一次进度更新
            if i % 10 == 0:
                sse_print("training_progress", {
                    "epoch": epoch + 1,
                    "step": i,
                    "total_steps": len(train_loader),
                    "current_norm": norm.item(),
                    "progress_percent": (i / len(train_loader)) * 100
                })
        
        sse_print("epoch_complete", {
            "epoch": epoch + 1,
            "norm": norm.item(),
            "message": f"norm: {norm.item()}"
        })

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            sse_print("early_stopping", {
                "message": "Early stopping triggered",
                "min_norm": min_norm.item(),
                "min_norm_count": min_norm_count
            })
            break

    return trigger.cpu(), mask.cpu()



def reverse_engineer():
    param = {
        "dataset": "cifar10",
        "Epochs": 10,
        "batch_size": 64,
        "lamda": 0.01,
        "num_classes": 10,
        "image_size": (32, 32)
    }
    sse_print("reverse_engineer_start", {"message": "Starting reverse engineering process"})
    
    model = torch.load('model_cifar10.pkl',weights_only=False).to(device)
    _, _, x_test, y_test = get_data(param)
    x_test = x_test[:100]
    y_test = y_test[:100]
    x_test, y_test = torch.from_numpy(x_test)/255., torch.from_numpy(y_test)
    train_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=param["batch_size"], shuffle=False)

    norm_list = []
    for label in range(param["num_classes"]):
        trigger, mask = train(model, label, train_loader, param)
        norm_list.append(mask.sum().item())
        
        # 保存触发器和掩码图像
        trigger_np = trigger.cpu().detach().numpy()
        trigger_np = np.transpose(trigger_np, (1,2,0))
        
        mask_np = mask.cpu().detach().numpy()
        
        # 发送完成信息
        sse_print("label_processing_complete", {
            "label": label,
            "mask_norm": mask.sum().item(),
            "message": f"Completed processing label {label}"
        })

    sse_print("reverse_engineer_complete", {
        "norm_list": norm_list,
        "message": "Reverse engineering completed",
        "final_norms": str(norm_list)
    })





if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sse_print("device_selected", {
        "device": str(device),
        "message": f"Using device: {device}"
    })
    reverse_engineer()
    sse_print("program_end", {"message": "Program execution completed"})