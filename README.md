# 共享词嵌入续写训练 · SGD 单步对照校验

面向文本模型训练工程师的**内存节省**实现与**一致性对照**工具：

* 同一个 float64 权重矩阵 `W`（种子 → 标准正态 → ×倍率）既作**输入嵌入**（查表），
  又作**输出投影**（`logits = h · Wᵀ`），即 weight tying。
* 损失为全部**有效**目标（目标号 `-100` 忽略）上的**平均交叉熵**，
  只做**一次** lr = 0.01 的 SGD。
* 自定义 `torch.autograd.Function` 计算损失：
  * 投影临时张量的**词表轴 ≤ 128**（`MAX_VOCAB_CHUNK`）；
  * 前向**不保存**任何分块分数（logits）或概率，只保存共享 W、隐状态、目标号与
    有效行索引；反向按块**重算** logits 与 `(max, sum-exp)` 归一化再求梯度；
  * 投影按 16 个嵌入维度固定顺序累加，单个 `[token, vocab]` 结果与批次大小、
    词表分块位置逐位无关（详见“实现要点”）；
  * 微批损失按各自有效 token 数加权后相加，等价于全局有效 token 平均。
* 独立的**稠密参考**实现不受限（完整 `[tokens, vocab]` logits + `log_softmax`），
  与分块实现共用同一个数学上等价的投影表达式。
* 网页录入 1–4 个微批（总计 ≤128 token），显示**损失 / 共享 W 梯度 / 更新后权重**
  三项对照误差，全部 ≤1e-8 才显示“通过”；同时验证把同一 token 流重新拆成
  1–4 个微批时损失、梯度、更新后权重不变，且每一路拆分仍与稠密参考一致。
* 内置四个样例：重复 token、不等有效长度（含整批被忽略）、大分数（scale=100，
  词表 4096）、大分数近似平局（种子 1587 的 `1931,3209` 报告用例）。

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

服务提供：

| 路径 | 说明 |
| --- | --- |
| `GET /` | 录入界面 |
| `GET /healthz` | 健康检查（Compose healthcheck 使用） |
| `GET /api/samples` | 三个内置样例 |
| `POST /api/run` | 运行对照，返回 JSON 报告 |

### 本地运行

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

> 运行环境为 CPU 版 PyTorch（镜像中显式安装 `--index-url .../whl/cpu`）。
> 本项目用到的都是 float64 CPU 算子，但若本机默认装的是 CUDA 版 PyTorch，
> 数值结果一致。

## 输入格式

```json
{
  "seed": 20240517,
  "scale": 1.0,
  "vocab_size": 256,
  "microbatches": [
    {"name": "dup-A", "tokens": [[4, 4], [4, 4], [7, 4], [4, 7]]},
    {"name": "dup-B", "tokens": [[4, 4], [9, 9], [9, -100], [4, 9]]}
  ]
}
```

* `seed`：0 … 2⁶³−1 的整数；`torch.Generator` 固定种子生成
  `randn(vocab_size, 16, dtype=float64)`。
* `scale`：0.1 … 100（权重倍率）。
* `vocab_size`：256 … 4096；嵌入维度固定 16。
* `microbatches`：1 … 4 个；每项 `tokens` 为 `[输入号, 目标号]` 列表。
  输入号必须落在词表内；目标号为 `-100` 或词表内整数。
* 全部微批的 token 总数（含 `-100` 项）≤ 128。
* 若**所有**目标都是 `-100`，请求返回 422 且不做任何参数更新
  （损失无定义，拒绝运行）。
* 单微批简写：顶层直接给 `"tokens": [...]` 也可。

## 对照报告

`POST /api/run` 返回（节选）：

```jsonc
{
  "ok": true,
  "tolerance": 1e-8,
  "chunked": { "loss": 0.31456133502, "grad_norm": 63.94, "weight_norm": 63.94 },
  "dense":   { "loss": 0.31456133503, "grad_norm": 63.94, "weight_norm": 63.94 },
  "errors":  { "loss_abs_diff": 4.5e-12, "grad_max_abs_diff": 2.8e-14,
               "weight_max_abs_diff": 1.4e-14 },
  "split":   { "max_loss_diff": 0.0, "max_grad_diff": 1.4e-14,
               "max_weight_diff": 0.0, "max_vs_dense": { ... }, "runs": [ ... ] },
  "checks":  { "loss": true, "grad": true, "updated_weight": true,
               "split_loss": true, "split_grad": true, "split_weight": true,
               "split_vs_dense_loss": true, "split_vs_dense_grad": true,
               "split_vs_dense_weight": true },
  "memory":  { "max_vocab_chunk": 128, "scores_or_probabilities_saved": false,
               "recompute_in_backward": true, ... }
}
```

* 三项主对照：`loss_abs_diff`、`grad_max_abs_diff`、`weight_max_abs_diff`
  （分块实现 vs 稠密参考）。
* 拆分不变量：把同一 token 流（保持顺序）重新均分为 1–4 个微批，各自跑一遍，
  与用户录入的微批结果比较，再取最大值；同时要求每一路重新拆分的结果仍与
  稠密参考相差 ≤1e-8。
* 只有九个 `checks` 全为 true，`ok` 才为 true，网页才显示“通过”。

## 实现要点

```
app/engine.py       校验、W 生成、自定义 autograd、稠密参考、对照报告
app/main.py         FastAPI 路由
app/static/index.html  录入与结果界面（无外部依赖）
app/samples.py      四个样例（重复 token、不等有效长度、大分数、大分数近似平局）
tests/test_engine.py   数值/约束/内存行为测试
tests/test_api.py      HTTP 层测试
```

自定义损失的核心：

```python
class ChunkedSoftmaxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, hidden, target):
        # 1) 逐词表块算 logits = project(h, W[chunk])（词表轴 ≤128），
        #    流式维护 (max, sum exp(logit-max)) 作为全局归一化；
        # 2) 只按块取目标 token 的 logit；
        # 3) loss = mean(max + log(sum) - target_logit)；
        # 4) 只保存 weight / hidden / target / valid_idx。
    @staticmethod
    def backward(ctx, grad_output):
        # 用保存的 hidden 与 W 重算每块 logits，按同一 (max, sum) 归一化，
        # 就地减去 one-hot 后按块累加 dW[start:end] += probs.T @ h，
        # dH += probs @ W[start:end]；不缓存前向的分数或概率。
```

* 重复输入号通过 `index_add_` 累加 `dH`，与 autograd 的 `index_select` 语义一致。
* **分块归一化用 `(max, sum exp(z-max))`，而不是先算 `lse` 再 `exp(z-lse)`。**
  `scale=100` 时 logits 约 3e5，一个 float64 ulp 约 6e-11；`z - lse` 会发生
  大数相消，把 softmax 概率的精度从 ~1e-16 降到 ~1e-11，再放大到梯度就是
  报告中的 ~1e-8 误差。`z - max` 因为 max 本身就是某个 logit，可以精确相减。
* **投影 `_project` 固定按 16 个嵌入维度顺序累加，不用 BLAS GEMM。**
  BLAS 的 `h @ W.T` 会按行数选择不同的分块/累加路径：同一个 token 单独成批
  和与三个重复行同批时，logits 末位不同；在 3e5 量级上这就是 1 ulp 的抖动，
  足以让 2+2 拆分与整批的梯度相差 64 以上（实测）。固定顺序后，单个
  `[token, vocab]` 的值与批次大小、词表分块位置**逐位**无关。
* `_project` 与 `(max, sum)` 归一化是**两条独立的必要条件**：只改其一，
  报告中的四个重复行拆分仍会超差。稠密参考也使用同一个 `_project`
  （数学上与原式完全相同），这样对照检验的是分块与微批拆分逻辑本身，
  而不是 BLAS 的形状相关舍入。
* 微批按“有效 token 数 / 全局有效 token 数”加权，因此拆分数不改变结果。
* 这些矩阵很小（≤128 行 × 16 列），torch 默认多线程会因线程调度开销慢 30 倍，
  引擎把 `torch.set_num_threads` 默认钉为 1（可用 `TORCH_NUM_THREADS` 覆盖）。

## 测试

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install -r requirements.txt      # 运行服务
python3 -m pip install -r requirements-dev.txt  # 运行测试（含 pytest/httpx）
python3 -m pytest tests -q
```

覆盖：种子可复现、权重共享（与手工 tying 图对比）、`-100` 忽略、全忽略拒绝、
边界与超限校验、128 token 上限、微批拆分不变量、重复 token 梯度累加、
有限差分梯度校验、大分数数值精度、`save_for_backward` 不含分数/概率、词表分块 ≤128、
投影的批次/分块逐位不变性、报告用例（种子 1587、词表 4096、倍率 100）的
单行 / 加忽略项 / 四行重复 / 重新拆分回归、HTTP 路由与样例端到端通过。

## 已知限制

* 仅 CPU、float64、单步 SGD；没有优化器状态、学习率调度或梯度裁剪。
* `scale` 在 W 生成时一次性乘上，等价于把整体 logits 放大；不做损失缩放。
* 网页不做并发限流，单次请求为同步计算；最重的输入（128 token、词表 4096、
  scale=100）单次约 3 秒，其余输入通常在 1 秒内。
* 对照误差阈值固定 1e-8（float64 下实测：损失约 4e-12，梯度与权重约 1e-14）。
* `_project` 为逐维累加，比 BLAS 慢（128×4096 约 30 ms），换来与批次形状无关的
  逐位一致；若将来词表/维度大幅增大，需要重新评估这条取舍。
