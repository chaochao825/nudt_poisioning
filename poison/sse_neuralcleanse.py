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
from utils import sse_envelope, RunSummary, set_summary_writer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def emit(event: str, data: dict, progress: float = None, log: str = None):
    message = data.get("message", event)
    details = {k: v for k, v in data.items() if k != "message"}
    return sse_envelope(
        event=event,
        progress=progress,
        message=message,
        log=log,
        details=details,
    )

def train(model, target_label, train_loader, param):
    emit("processing_label", {"label": target_label, "message": f"Processing label: {target_label}"}, progress=15)

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
                emit("training_progress", {
                    "epoch": epoch + 1,
                    "step": i,
                    "total_steps": len(train_loader),
                    "current_norm": norm.item(),
                    "progress_percent": (i / len(train_loader)) * 100
                }, progress=40)
        
        emit("epoch_complete", {
            "epoch": epoch + 1,
            "norm": norm.item(),
            "message": f"norm: {norm.item()}"
        }, progress=55)

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            emit("early_stopping", {
                "message": "Early stopping triggered",
                "min_norm": min_norm.item(),
                "min_norm_count": min_norm_count
            }, progress=60)
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
    emit("reverse_engineer_start", {"message": "Starting reverse engineering process"}, progress=5)
    
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
        emit("label_processing_complete", {
            "label": label,
            "mask_norm": mask.sum().item(),
            "message": f"Completed processing label {label}"
        }, progress=75)

    final_payload = emit("final_result", {
        "norm_list": norm_list,
        "message": "Reverse engineering completed",
        "final_norms": str(norm_list)
    }, progress=100)
    return final_payload





if __name__ == "__main__":
    summary_writer = RunSummary("./output/neuralcleanse_defense", filename="defense_summary.json")
    set_summary_writer(summary_writer)
    final_payload = None
    emit("device_selected", {
        "device": str(device),
        "message": f"Using device: {device}"
    })
    try:
        final_payload = reverse_engineer()
        emit("program_end", {"message": "Program execution completed"}, progress=100)
    finally:
        summary_writer.flush(extra={"final_event": final_payload})
        set_summary_writer(None)