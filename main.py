import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from utils import ensure_path, sse_envelope, sse_print, default_callback_params


PRETRAINED_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"


def type_switch(environ_value, value):
    if isinstance(value, int):
        return int(environ_value)
    if isinstance(value, float):
        return float(environ_value)
    if isinstance(value, bool):
        # Env vars arrive as strings; consider truthy strings as True.
        return str(environ_value).lower() in {"1", "true", "yes", "y"}
    return environ_value


def parse_args():
    parser = argparse.ArgumentParser(description="BadNet style model poisoning demo")
    parser.add_argument("--input_path", type=str, default="../input", help="input path")
    parser.add_argument("--output_path", type=str, default="../output", help="output path")
    parser.add_argument(
        "--method",
        type=str,
        default="model_poisoning",
        choices=["model_poisoning", "dynamic_backdoor", "physical_backdoor"],
        help="which poisoning/backdoor variant to run",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classify",
        choices=["classify"],
        help="task type; only classification is supported for this demo",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="cifar10",
        choices=["cifar10"],
        help="dataset name; only cifar10 is supported for this demo",
    )
    parser.add_argument("--epochs", type=int, default=1, help="training epochs")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument(
        "--poison_rate",
        type=float,
        default=0.1,
        help="fraction of training data to poison",
    )
    parser.add_argument(
        "--target_label", type=int, default=0, help="target label for backdoor"
    )
    parser.add_argument(
        "--trigger_size", type=int, default=3, help="square trigger size in pixels"
    )
    parser.add_argument(
        "--train_subset",
        type=int,
        default=2000,
        help="subset size for quick demo training (0 = full dataset)",
    )
    parser.add_argument(
        "--test_subset",
        type=int,
        default=500,
        help="subset size for quick demo evaluation (0 = full dataset)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device id (e.g., 0 for cuda:0, or 'cpu')",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="dataloader workers"
    )

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict_environ = {}
    for key, value in args_dict.items():
        env_key = key.upper()
        args_dict_environ[key] = type_switch(os.getenv(env_key, value), value)
    return argparse.Namespace(**args_dict_environ)


def emit(
    event: str,
    progress,
    message: str,
    log: Optional[str] = None,
    details: Optional[dict] = None,
    resp_code: int = 0,
    resp_msg: str = "操作成功",
):
    sse_envelope(
        event=event,
        progress=progress,
        message=message,
        log=log,
        details=details,
        resp_code=resp_code,
        resp_msg=resp_msg,
        callback_params=default_callback_params(),
    )


def validate_support(args):
    """
    Ensure the requested task/data/method are supported; emit clear SSE failure if not.
    """
    supported_tasks = {"classify"}
    supported_data = {"cifar10"}
    supported_methods = {"model_poisoning", "dynamic_backdoor", "physical_backdoor"}

    if args.task not in supported_tasks:
        msg = f"Unsupported task '{args.task}'. Supported: {sorted(supported_tasks)}"
        emit("config_validation", progress=0, message=msg, resp_code=1, resp_msg="操作失败")
        raise ValueError(msg)
    if args.data not in supported_data:
        msg = f"Unsupported dataset '{args.data}'. Supported: {sorted(supported_data)}"
        emit("config_validation", progress=0, message=msg, resp_code=1, resp_msg="操作失败")
        raise ValueError(msg)
    if args.method not in supported_methods:
        msg = f"Unsupported method '{args.method}'. Supported: {sorted(supported_methods)}"
        emit("config_validation", progress=0, message=msg, resp_code=1, resp_msg="操作失败")
        raise ValueError(msg)
    emit(
        "config_validation",
        progress=5,
        message="Configuration validated.",
        log="[5%] 参数校验通过",
        details={"task": args.task, "data": args.data, "method": args.method},
    )


def prepare_device(device_flag: str) -> torch.device:
    if device_flag == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
        emit("device_selected", progress=None, message="Using CPU.")
        return device
    try:
        device_index = int(device_flag)
        device = torch.device(f"cuda:{device_index}")
    except ValueError:
        device = torch.device(device_flag)
    emit("device_selected", progress=None, message=f"Using device {device}.")
    return device


@dataclass
class TriggerConfig:
    size: int
    color: Tuple[int, int, int]
    position: str
    dynamic: bool = False


class TriggerApplier:
    def __init__(self, cfg: TriggerConfig):
        self.cfg = cfg

    def _choose_position(self, width: int, height: int, dynamic: bool) -> Tuple[int, int]:
        if dynamic or self.cfg.dynamic:
            x = random.randint(0, max(0, width - self.cfg.size))
            y = random.randint(0, max(0, height - self.cfg.size))
            return x, y

        if self.cfg.position == "top_left":
            return 0, 0
        if self.cfg.position == "center":
            return (width - self.cfg.size) // 2, (height - self.cfg.size) // 2
        # default bottom-right
        return max(0, width - self.cfg.size), max(0, height - self.cfg.size)

    def apply(self, image, dynamic: bool = False):
        """
        Add a square trigger to a PIL image.
        """
        img = image.copy()
        width, height = img.size
        start_x, start_y = self._choose_position(width, height, dynamic)
        pixels = img.load()
        for dx in range(self.cfg.size):
            for dy in range(self.cfg.size):
                x = min(width - 1, start_x + dx)
                y = min(height - 1, start_y + dy)
                pixels[x, y] = self.cfg.color
        return img


class BadNetsDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        transform,
        injector: TriggerApplier,
        poison_rate: float,
        target_label: int,
        dynamic_trigger: bool = False,
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        self.injector = injector
        self.target_label = target_label
        self.dynamic = dynamic_trigger

        total = len(base_dataset)
        poison_count = int(poison_rate * total)
        poison_count = max(1, poison_count) if poison_rate > 0 else 0
        self.poison_indices = set(random.sample(range(total), poison_count)) if poison_count else set()
        emit(
            "poison_generation_start",
            progress=20,
            message="开始生成投毒样本",
            log="[20%] 选择投毒样本并配置触发器",
            details={
                "poisoning_phase": "sample_generation",
                "samples_to_poison": poison_count,
                "total_samples": total,
                "poisoning_ratio": poison_rate,
                "target_class": target_label,
                "trigger_pattern": "square",
            },
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        poisoned = index in self.poison_indices
        if poisoned:
            image = self.injector.apply(image, dynamic=self.dynamic)
            label = self.target_label
        if self.transform:
            image = self.transform(image)
        return image, label, poisoned


class AlwaysTriggeredDataset(Dataset):
    def __init__(self, base_dataset: Dataset, transform, injector: TriggerApplier, target_label: int):
        self.base_dataset = base_dataset
        self.transform = transform
        self.injector = injector
        self.target_label = target_label

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, _ = self.base_dataset[index]
        image = self.injector.apply(image, dynamic=True)
        label = self.target_label
        if self.transform:
            image = self.transform(image)
        return image, label, True


def load_pretrained_vgg16(model_dir: str, num_classes: int, device: torch.device) -> nn.Module:
    ensure_path("model_dir_ready", model_dir, create=True)
    emit(
        "model_loaded",
        progress=10,
        message="目标模型加载中",
        log="[10%] 开始下载/加载 VGG16 权重",
        details={
            "model_name": "VGG16",
            "model_path": model_dir,
            "model_type": "classification",
        },
    )
    state_dict = torch.hub.load_state_dict_from_url(
        PRETRAINED_URL,
        model_dir=model_dir,
        progress=True,
        map_location="cpu",
    )
    model = models.vgg16()
    model.load_state_dict(state_dict)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model.to(device)
    emit(
        "model_loaded",
        progress=12,
        message="目标模型加载成功",
        log="[12%] VGG16 加载完成并替换分类头",
        details={
            "model_name": "VGG16",
            "model_path": os.path.join(model_dir, os.path.basename(PRETRAINED_URL)),
            "model_type": "classification",
            "input_shape": [3, 224, 224],
            "num_classes": num_classes,
        },
    )
    return model


def split_dataset(dataset: Dataset, subset_size: int) -> Dataset:
    if subset_size and subset_size < len(dataset):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        selected = indices[:subset_size]
        return Subset(dataset, selected)
    return dataset


def prepare_dataloaders(
    args,
    injector: TriggerApplier,
    dynamic_trigger: bool,
    target_label: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if args.data != "cifar10":
        msg = f"Dataset '{args.data}' is not supported for task '{args.task}'."
        emit("dataset_validation", progress=0, message=msg, resp_code=1, resp_msg="操作失败")
        raise ValueError(msg)

    dataset_root = os.path.join(args.input_path, "data", args.data)
    ensure_path("dataset_path_ready", dataset_root, create=True)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    base_train = datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=None)
    base_test = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=None)

    base_train = split_dataset(base_train, args.train_subset)
    base_test = split_dataset(base_test, args.test_subset)

    poisoned_train = BadNetsDataset(
        base_dataset=base_train,
        transform=transform,
        injector=injector,
        poison_rate=args.poison_rate,
        target_label=target_label,
        dynamic_trigger=dynamic_trigger,
    )
    clean_test = BadNetsDataset(
        base_dataset=base_test,
        transform=transform,
        injector=injector,
        poison_rate=0.0,
        target_label=target_label,
        dynamic_trigger=False,
    )
    backdoor_test = AlwaysTriggeredDataset(
        base_dataset=base_test,
        transform=transform,
        injector=injector,
        target_label=target_label,
    )

    train_loader = DataLoader(
        poisoned_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    clean_loader = DataLoader(
        clean_test,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    backdoor_loader = DataLoader(
        backdoor_test,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    emit(
        "dataset_loaded",
        progress=15,
        message="数据集加载完成",
        log="[15%] 数据集加载完成，准备投毒/测试集",
        details={
            "dataset_name": args.data,
            "dataset_path": dataset_root,
            "train_batches": len(train_loader),
            "clean_batches": len(clean_loader),
            "backdoor_batches": len(backdoor_loader),
        },
    )

    emit(
        "poison_generation_completed",
        progress=40,
        message="投毒样本生成完成",
        log="[40%] 投毒样本生成完成",
        details={
            "poisoning_phase": "sample_generation",
            "samples_generated": len(poisoned_train.poison_indices),
            "trigger_pattern": "square",
            "target_class": target_label,
            "poisoned_dataset_size": len(poisoned_train),
        },
    )
    return train_loader, clean_loader, backdoor_loader


def train_one_epoch(
    epoch: int,
    total_epochs: int,
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets, poisoned_flags) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % max(1, len(loader) // 5) == 0:
            progress = round((batch_idx + 1) / len(loader) * 100, 2)
            emit(
                "poison_training_progress",
                progress=45 + progress * 0.4 / 100,  # keep around 45-85 during training
                message=f"Epoch {epoch}/{total_epochs} 训练中",
                log=f"[{progress:.1f}%] 训练批次 {batch_idx+1}/{len(loader)}",
                details={
                    "current_epoch": epoch,
                    "total_epochs": total_epochs,
                    "batch": batch_idx + 1,
                    "total_batches": len(loader),
                    "training_loss": float(loss.item()),
                    "poisoned_in_batch": int(sum(poisoned_flags)),
                },
            )
    return running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, expect_target: Optional[int] = None) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        if expect_target is None:
            correct += (preds == targets).sum().item()
        else:
            target_tensor = torch.full_like(preds, expect_target)
            correct += (preds == target_tensor).sum().item()
        total += inputs.size(0)
    return correct / max(1, total)


def select_trigger_config(args) -> TriggerConfig:
    if args.method == "dynamic_backdoor":
        return TriggerConfig(
            size=max(2, args.trigger_size),
            color=(255, 255, 0),
            position="bottom_right",
            dynamic=True,
        )
    if args.method == "physical_backdoor":
        return TriggerConfig(
            size=max(6, args.trigger_size),
            color=(255, 0, 0),
            position="center",
            dynamic=False,
        )
    return TriggerConfig(
        size=args.trigger_size,
        color=(255, 255, 255),
        position="bottom_right",
        dynamic=False,
    )


def run_badnet(args):
    random.seed(42)
    torch.manual_seed(42)

    validate_support(args)

    ensure_path("input_path_validated", args.input_path, create=True)
    ensure_path("output_path_validated", args.output_path, create=True)

    emit(
        "process_start",
        progress=5,
        message="开始数据投毒攻击任务",
        log="[5%] 开始 BadNets 数据投毒任务",
        details={
            "attack_method": "BadNets",
            "target_model": "VGG16",
            "dataset": args.data,
            "poisoning_ratio": args.poison_rate,
        },
    )

    device = prepare_device(str(args.device))

    trigger_cfg = select_trigger_config(args)
    injector = TriggerApplier(trigger_cfg)

    train_loader, clean_loader, backdoor_loader = prepare_dataloaders(
        args=args,
        injector=injector,
        dynamic_trigger=trigger_cfg.dynamic or args.method == "dynamic_backdoor",
        target_label=args.target_label,
    )

    model = load_pretrained_vgg16(
        model_dir=os.path.join(args.input_path, "model"),
        num_classes=10,
        device=device,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    history: List[dict] = []
    emit(
        "poison_training_start",
        progress=45,
        message="开始投毒模型训练",
        log="[45%] 使用投毒数据集开始训练",
        details={
            "training_phase": "poison_training",
            "total_epochs": args.epochs,
            "batch_size": args.batch,
            "learning_rate": 0.001,
        },
    )
    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(
            epoch=epoch,
            total_epochs=args.epochs,
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        clean_acc = evaluate(model, clean_loader, device)
        asr = evaluate(model, backdoor_loader, device, expect_target=args.target_label)
        record = {
            "epoch": epoch,
            "loss": epoch_loss,
            "clean_acc": clean_acc,
            "asr": asr,
        }
        history.append(record)
        emit(
            "epoch_completed",
            progress=45 + epoch / max(1, args.epochs) * 40,
            message=f"第{epoch}轮训练完成",
            log=f"[epoch {epoch}] loss={epoch_loss:.4f}, clean_acc={clean_acc:.4f}, asr={asr:.4f}",
            details={
                "current_epoch": epoch,
                "total_epochs": args.epochs,
                "training_loss": round(epoch_loss, 4),
                "training_accuracy": round(clean_acc, 4),
                "backdoor_success_rate": round(asr * 100, 2),
                "learning_rate": 0.001,
            },
        )

    ensure_path("output_model_dir_ready", args.output_path, create=True)
    ckpt_path = os.path.join(args.output_path, f"{args.method}_vgg16_badnet.pth")
    torch.save(model.state_dict(), ckpt_path)
    metrics_path = os.path.join(args.output_path, f"{args.method}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    emit(
        "artifact_saved",
        progress=90,
        message="模型与指标保存完成",
        log="[90%] 模型与指标已保存",
        details={
            "model_path": ckpt_path,
            "metrics_path": metrics_path,
        },
    )
    final_clean = history[-1]["clean_acc"]
    final_asr = history[-1]["asr"]
    emit(
        "run_completed",
        progress=100,
        message="投毒任务完成",
        log="[100%] 任务完成",
        details={
            "final_clean_acc": round(final_clean, 4),
            "final_attack_success_rate": round(final_asr * 100, 2),
        },
    )


def main():
    args = parse_args()
    sse_print(
        "config",
        {
            "status": "success",
            "message": "Parsed arguments.",
            "args": vars(args),
        },
    )
    run_badnet(args)


if __name__ == "__main__":
    main()

