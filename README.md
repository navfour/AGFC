# AG-FC: Adaptive Graph Forecasting for Scientific Topic Popularity

基于论文 *Learning Adaptive Graphs to Uncover Potential Relationships for Scientific Topic Popularity Trend Forecasting* 的代码实现与实验复现仓库。  
核心入口脚本为 `dynamic_run_snapshot.py`，用于在多种图建模模式下训练/评估 AG-FC（Adaptive Graph Fully Connected）模型。

## 1. 项目目标

本项目将“科研主题热度预测”建模为**图结构感知的多变量时间序列预测**任务：

- 节点：主题或子领域（取决于数据集粒度）
- 时间：年度 publication volume（1975--2024）
- 预测：给定历史窗口，预测未来 `H` 步（论文主实验为 `H=1`）

论文中的关键结论：

- AG-FC 在主实验中获得 **12/24** 项 Top-1。
- 在更细粒度 subfield 数据上，AG-FC 获得 **11/12** 项 Top-1。
- 相比 FC-GAGA 与 FC-GAGA（without graph gate），AG-FC 分别在 **20/24**、**21/24** 设置上更优。

## 2. 方法与代码对应

`dynamic_run_snapshot.py` 支持四种图模式（`--graph_mode`）：

- `independent`：不使用图结构（节点独立建模）
- `basegraph`：仅使用基图（由时间窗口内快照融合）
- `adaptiveonly`：仅使用自适应图（从可学习节点嵌入构建）
- `hybrid`：融合基图与自适应图（支持 SVD 初始化）

与论文方法设计的代码映射如下：

- 动态快照图与滑窗融合：`dynamic/dynamic_graph_provider.py`
- 自适应邻接学习与 hybrid 融合：`dynamic/dynamic_adaptive_layers.py`
- FC-GAGA 主干与图算子（GCN/GAT/GraphSAGE）：`dynamic/dynamic_model_snapshot.py`
- 数据切分与样本构造：`dynamic/dynamic_generate_training_data.py`

## 3. 框架图（已迁移到非临时目录）

> 为避免 `newlatex/` 删除后 README 失效，所有 README 用图已复制到 `docs/readme_assets/`。

![AG-FC framework](docs/readme_assets/agfc_framework.png)

## 4. 数据组织

当前仓库内置 8 个 OpenAlex 子集（4 个 domain + 4 个 subfield）：

- `data/oa_1`, `data/oa_2`, `data/oa_3`, `data/oa_4`
- `data/oa_1702`, `data/oa_1705`, `data/oa_1707`, `data/oa_1710`

每个子集目录包含：

- `*_covert.csv`：节点年度时间序列（行是年份，列是节点）
- `*_coo_pkl/`：每年的共现邻接矩阵快照（`{subset}_cooccur_{year}.pkl`）
- `*_coo/`：对应 Excel 版本（分析用）

默认参数使用 `oa_2`。

## 5. 环境依赖

建议使用 `Python 3.10/3.11`（避免环境架构混装问题），核心依赖：

- `tensorflow`
- `numpy`
- `pandas`
- `gdown`

可参考：

```bash
pip install tensorflow numpy pandas gdown
```

## 6. 快速开始

### 6.1 默认运行（oa_2）

```bash
python dynamic_run_snapshot.py
```

### 6.2 复现实验常用配置：Adaptive Only（无 SVD）

```bash
python dynamic_run_snapshot.py \
  --data_csv data/oa_1702/oa_1702_covert.csv \
  --adjacency_dir data/oa_1702/oa_1702_coo_pkl \
  --adjacency_pattern "1702_cooccur_{year}.pkl" \
  --graph_mode adaptiveonly \
  --adaptive_init random \
  --graph_layer_type graphsage \
  --history_length 5 \
  --horizon 1
```

### 6.3 Hybrid + SVD（兼顾精度与先验一致性）

```bash
python dynamic_run_snapshot.py \
  --data_csv data/oa_1702/oa_1702_covert.csv \
  --adjacency_dir data/oa_1702/oa_1702_coo_pkl \
  --adjacency_pattern "1702_cooccur_{year}.pkl" \
  --graph_mode hybrid \
  --adjacency_fusion mean \
  --adaptive_init mean \
  --graph_layer_type graphsage \
  --history_length 5 \
  --horizon 1
```

## 7. 常用参数

| 参数 | 说明 | 典型值 |
|---|---|---|
| `--graph_mode` | 图建模模式 | `independent` / `basegraph` / `adaptiveonly` / `hybrid` |
| `--adjacency_fusion` | 基图快照融合方式 | `last` / `sum` / `mean` / `learned_decay` |
| `--adaptive_init` | 自适应图初始化方式 | `last` / `sum` / `mean` / `random` |
| `--graph_layer_type` | 图编码器类型 | `none` / `gcn` / `gat` / `graphsage` |
| `--history_length` | 历史窗口长度 | 默认 `5` |
| `--horizon` | 预测步长 | 默认 `1` |
| `--apt_size` | 自适应邻接低秩维度 | 默认来自超参配置（常用 `10`） |
| `--use_time_gate` | 是否启用时间门控 | 默认关闭 |

## 8. 输出结果

每次运行会在 `dynamic_results0130/` 下创建单独实验目录，典型产物包括：

- `config.json`：运行参数与超参
- `metrics.json`：总体指标与分 horizon 指标
- `prediction_by_node_year.csv`：逐节点逐年份预测
- `node_error_summary.csv`：节点级误差统计
- `y_true.npy` / `y_pred.npy`：标签与预测张量
- `adaptive_adj.npy` / `adaptive_adj_pure.npy`：融合图与纯自适应图（使用 adaptive 模式时）
- `base_graph.npy` / `base_graph_raw.npy`：基图（hybrid/base 模式相关）
- `best_weights.weights.h5`：最佳权重
- `dynamic_saved_model/`：导出模型

同时会在 `dynamic_results0130/results_index.csv` 中追加本次实验索引。

## 9. 论文结果摘要（README 可追踪数据）

为避免依赖临时论文目录，以下摘要数据已固化到：

- `docs/readme_assets/paper_result_summary.csv`
- `docs/readme_assets/paper_main_table_top1_by_model.csv`

主实验 Top-1 统计（Table 5 汇总）：

| Model | Top-1 Count |
|---|---:|
| AG-FC | 12 |
| SARIMAX | 8 |
| FC-GAGA | 3 |
| LSTM | 1 |
| FC-GAGA (without graph gate) | 0 |
| NeuralProphet | 0 |
| Informer | 0 |
| TimeX++ | 0 |

此外，GraphSAGE 在图算子消融中表现最稳健（Top-1: 16/24，Top-3: 52/72）。

## 10. 结果示意图（已迁移）

Subfield 1702 五年趋势预测示意：

![Forecast trend oa1702](docs/readme_assets/forecast_trend_oa1702.png)

Subfield 1702 拓扑转变（base → adaptive → fused）：

![Topology transition oa1702](docs/readme_assets/topology_transition_oa1702.png)

## 11. 引用

如果本仓库对你的研究有帮助，请引用对应论文：

```text
Learning Adaptive Graphs to Uncover Potential Relationships for Scientific Topic Popularity Trend Forecasting
Changwang Li, Xuecan Tian, Zeyu Deng, Jin Mao
```

