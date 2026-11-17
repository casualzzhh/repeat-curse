# FlashDLM 时空特征抗重复实验

这是截图对应版本的**完整实验代码**：LoRA 微调 Dream 主干，利用连续去噪隐藏状态的空间/时间卷积预测重复风险，构造辅助抗重复损失，并在推理时执行风险门控的 remask 和 AR 引导修复。

包含数据准备、固定投影、原生自由生成轨迹、训练、开发集阈值选择、原始/完整模型评估、原文审查和证据导出。该版本为 **96 次更新、48 条自由轨迹、两组各32条输出**的小规模实验。

**研究结论：有局部改善，尚不能认定整体有效。** 这不是一个已经证明有效的抗重复方案。数据、阈值和解码策略均保留截图版本的语义；后来的全阈值实验和“Dream/AR 严格一致才提交”的设想不属于此版本。

## 参考结果

|指标|原始模型|完整方法|
|---|---:|---:|
|明显循环回答，人工审查/28|5|3|
|数学答案匹配/12|10|10|
|指定重复指令严格匹配/4|3|1|
|平均生成秒/主任务|10.87|18.65|

4个旧循环消失，1个仍存在，并新增2例；另有1例语义重复不确定。人工审查不盲、单种子、提示已观察过。6次实际修复均提交了AR替代token，但token改变不代表语义修复成功。详见 [参考结果](docs/reference-results.md) 和 [已知局限](docs/known-issues.md)。

## 环境

- Linux、Python3.11、支持 BF16 的 CUDA GPU；参考实验使用一张24GB GPU。
- 默认要求至少22GiB可用显存；程序不会终止其他进程。
- 实测 Torch2.7.0+cu118、Transformers4.46.0；其他组合未验证。

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

先明确下载两个模型；仓库不包含模型权重。以下模型修订版对应参考实验：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Dream-org/Dream-v0-Instruct-7B', revision='05334cb9faaf763692dcf9d8737c642be2b2a6ae')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')"
```

运行默认离线。可复制 `config.example.json` 为 `config.local.json`，将模型配置为本机目录，运行前设置：

```bash
export REP_CONFIG="$PWD/config.local.json"
```

相对路径按仓库根目录解析。`REP_RUN_DIR`、`REP_ASSET_DIR`、`REP_DREAM_MODEL`、`REP_AR_MODEL` 等环境变量可覆盖同名配置。路径可变，模型架构与分词版本应保持不变；预编码数据和3584维投影不适用于任意其他模型。

## 一次执行完整流程

```bash
python prepare_assets.py
python -m unittest discover -s tests -v
export REP_RUN_DIR="$PWD/runs/pilot-001"
python pipeline.py
```

流水线依次执行：资产检查 → CPU测试 → 真实模型修复预检 → LoRA/CNN训练 → 开发集校准 → 原始模型32条 → 完整方法32条 → 自动报告。

全部生成结果和权重保存在 `REP_RUN_DIR`，与源码分开。首次训练前检查原始/关闭控制器/仅记录轨迹三者的完整192槽输出一致性；训练检查辅助损失对LoRA和4帧特征的非零梯度。

数据已包含公开 GSM8K 的精确预编码划分：1024训练、64开发、64测试，以及152条提示库。`prepare_assets.py` 重建固定随机投影（seed501、3584×128），校验其张量SHA256。无需提交或下载旧服务器目录。

若希望从原始GSM8K文件核对划分，在模型已缓存后运行：

```bash
python prepare_assets.py --raw-train data/raw/train.jsonl --raw-test data/raw/test.jsonl
```

此命令核对原始文件哈希、源顺序与编码后的记录；不匹配会停止，避免将不同数据称为精确复现。原始数据来源和许可证见 `data/README.md`。

## 分阶段执行

```bash
python smoke.py
python experiment.py train
python experiment.py calibrate
python experiment.py original
python experiment.py complete
python report.py
python show_review.py 0 6
```

`train` 固定为参考实验的96次更新；每32次更新前生成16条新轨迹。常规扩散损失使用192个GSM训练样本，辅助损失从自由轨迹采样。自由轨迹没有金标准回答前缀。

完整方法包含推理修复；`complete` 不是仅训练增加loss的消融。该版本修复时记录Dream新预测，但最终由AR从允许候选中选取替代token，并不要求二者相等。

## 输出与人工审查

- `checkpoint.pt`：LoRA、CNN及更新数；不含基础模型，不含完整优化器状态。
- `training-complete.json`、`gradient-proof.json`：训练与梯度检查。
- `thresholds.json`：三个风险阈值及开发集统计。
- `eval/original/`、`eval/complete/`：每条完整原文、token与修复动作。
- `summary.json`、`paired-cases.json`、`REPORT.md`：自动统计。
- `multiscale-evidence.zip`、`manifest.json`：证据和逐文件哈希。

自动重复统计使用有已知缺陷的启发式，不能直接当成真实循环率。人工审查另做：

```bash
python review.py init
# 编辑 runs/pilot-001/manual-review.json，逐条填写 true/false；不确定保留 null
python review.py summarize
```

人工标签绑定原文SHA256，文本变化后拒绝复用。脚本不会自动产生参考实验的“5→3”；新输出必须重新审核。

训练中断的中间checkpoint没有优化器状态，应使用新的输出目录重新训练。已完成训练后的评估可用 `python pipeline.py --resume` 续跑已完成阶段；存在运行锁时先核对进程，勿直接并发覆盖。

续跑会核对代码、模型配置和数据签名，防止将不同版本的阶段混在一起。

## 代码结构

```text
settings.py / config.example.json  路径、模型修订版与离线配置
flash_experiment.py                最小FlashDLM兼容桥
backbone.py                       LoRA、隐藏状态和带检查点的前向
risk.py                           空间卷积＋因果时间卷积＋三类风险
repeats.py                        截图版本的重复规则
experiment.py                     轨迹、损失、训练、阈值和两组评估
sampling.py                       原生采样器接入、remask和AR修复
pipeline.py / prepare_assets.py   完整流程与资产重建
report.py / review.py             自动统计与人工审查
third_party/flash_dlm/            参考运行所需的最小底层源码快照
```

方法公式见 [METHOD.md](METHOD.md)。上游文件哈希见 `third_party/flash_dlm/UPSTREAM_FILES.json`；除可配置分词器路径的加载补丁，底层模型和采样数学保持快照行为。源码没有记录上游Git提交号，不将当前上游main冒充为实验提交。

## 上传 GitHub

只上传本目录。`.gitignore` 已排除本机配置、模型权重、运行产物、环境和压缩包。

```bash
git init
git add .
git commit -m "Add spatiotemporal repetition-control experiment"
git branch -M main
# 在GitHub创建空仓库后，添加自己的remote并push
```

第三方代码、数据与模型的来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 本次整理的验证范围

已在参考环境通过12项CPU测试、固定投影精确重建和真实7B修复预检。额外选择一条确实发生修复的双筒望远镜样本，公开版的原始/完整两组各192槽输出及AR动作与历史记录一致。详见 `VALIDATION.json`。

本次没有重新执行96步训练或重跑全部32对任务；上述参考结果来自已完成的原实验。

解压后可用不依赖PyTorch的命令检查源码包：

```bash
python tools/verify_manifest.py
```
