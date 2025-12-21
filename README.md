# 数据投毒与后门攻击（poisioning）

基于 VGG16 + CIFAR10 的简化 BadNet 训练示例，默认提供 3 个名称的攻击方式：
- `model_poisoning`：主流程，使用固定位置白色方块触发器。
- `dynamic_backdoor`：基于同一实现，触发器位置随机且颜色为黄，便于快速复用。
- `physical_backdoor`：同一实现的红色中心触发器，模拟物理粘贴效果。

其中 **model_poisoning** 已完整实现，可运行并输出大量 SSE 风格日志，其他两种是在其基础上的轻量修改。

## 运行
在 `nudt/poisioning` 目录执行：
```bash
python main.py \
  --input_path ../input \
  --output_path ../output \
  --method model_poisoning \
  --epochs 1 \
  --batch 32
```
或使用 SSE 对齐的多攻击模拟：
```bash
bash scripts/run_attack.sh BadNets   # 也可选 Trojan / FeatureCollision / Triggerless / DynamicBackdoor / PhysicalBackdoor / NeuronInterference / ModelPoisoning / FGSM / PGD / Optimization / MI-FGSM / BackdoorPoison
```

### Docker 快速测试
```bash
cd nudt/poisioning
docker build -t poisioning:latest .
IMAGE=poisioning:latest bash scripts/docker-test-attack.sh BadNets
```
所有参数都可通过同名大写环境变量覆盖（如 `METHOD`, `POISON_RATE` 等）。

## 输入/输出

- 输出：训练好的权重与指标保存在 `output/`，文件名包含所选 `method`。

## 日志风格
沿用 `@nudt` 项目的 SSE 输出格式，每个阶段都会打印 `event: xxx / data: {...}`，便于流式展示进度与指标。默认开启大量阶段与 epoch 级别日志。***

