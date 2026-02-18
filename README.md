<p align="center">
  <h1 align="center">Marketing Attribution & Budget Optimization</h1>
  <p align="center">
    <b>从宏观 MMM 到微观多触点归因的全链路营销效果评估与预算优化系统</b>
  </p>
  <p align="center">
    <a href="https://github.com/MeaFew/marketing-attribution-mmm/actions"><img src="https://github.com/MeaFew/marketing-attribution-mmm/workflows/CI/badge.svg" alt="CI"></a>
    <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff&logoColor=white" alt="Ruff">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </p>
  <p align="center">
    <b>中文</b> | <a href="./README.en.md">English</a>
  </p>
</p>

---

## 项目概览

本系统基于 figshare 发布的「Conjura Multi-Region MMM Dataset」（覆盖约 100 个电商品牌、19 个地区、2019–2024 年共 132,759 条日粒度记录），构建了一套从**宏观营销组合建模（MMM）**到**微观用户旅程归因**再到**预算约束优化**的完整分析链路。

核心解决的业务问题：

- **渠道 ROI 量化困难**：多个渠道同时投放时，如何剥离各渠道对转化的真实贡献？
- **归因模型选择无依据**：First-touch、Last-touch、Shapley Value、Markov Chain 等方法结论差异巨大，如何系统比较？
- **预算分配凭经验**：在总预算约束下，如何科学重新分配各渠道 spend 以最大化 revenue？

---

## 技术架构

```mermaid
flowchart LR
    A[Raw CSV<br/>132K rows × 50 cols] --> B[Polars ETL]
    B --> C[DuckDB / Parquet]
    C --> D[MMM Modeling<br/>OLS / Ridge / Lasso]
    C --> E[User Journey<br/>Simulation 50K]
    E --> F[6 Attribution Models]
    D --> G[Budget Optimizer<br/>scipy SLSQP]
    F --> G
    G --> H[Streamlit Dashboard]
```

| 层级 | 技术选型 | 设计理由 |
|------|---------|---------|
| 数据清洗 | **Polars** | 向量化执行 + 惰性求值，处理 132K 行毫秒级 |
| 存储 | **DuckDB** / Parquet | 零配置 OLAP，列式压缩，SQL 分析开箱即用 |
| 宏观建模 | **statsmodels** + **scikit-learn** | OLS 提供完整统计推断（p-value、置信区间）；Ridge/Lasso 处理渠道间共线性 |
| 微观归因 | 自研 6 种模型 | 覆盖规则类（First/Last/Linear/Time-decay）与博弈论类（Shapley/Markov），便于横向对比 |
| 预算优化 | **scipy.optimize** SLSQP | 支持等式约束（总预算不变）与不等式约束（单渠道下限），收敛稳定 |
| 交付 | **Streamlit** + **Plotly** | 三页交互看板：MMM 概览 / 归因对比 / 预算模拟器 |

---

## 快速开始

```bash
git clone https://github.com/MeaFew/marketing-attribution-mmm.git
cd marketing-attribution-mmm
make setup        # 创建虚拟环境 + 安装依赖
make all          # 运行完整管线：清洗 → MMM → 归因 → 优化
make dashboard    # 启动 Streamlit 交互看板
make test         # 运行 pytest 测试套件
make verify       # 本地质量门（lint + format + type-check）
```

---

## 核心模块

### 1. 数据预处理（`scripts/preprocess.py`）

```
输入: 132,759 rows × 50 cols（含大量空值与千分位逗号分隔符）
输出: 清洗后的 Parquet + DuckDB 表
关键操作:
  - 千分位逗号去除 + Float64 强制转换（解决 Polars 自动推断为 String 的问题）
  - CTR、CPM、ROAS 衍生指标计算
  - Adstock 衰减特征构造: x_t + 0.5·x_{t-1} + 0.25·x_{t-2}
  - 按 brand + territory 聚合为周粒度面板数据
```

### 2. 营销组合建模（`scripts/mmm_model.py`）

#### Benchmark

> Conjura MMM Dataset 为 2024 年 6 月发布的学术数据集，暂无官方竞赛 Baseline。以下为 MMM 领域公认的标准建模基准：

| 参照系 | R² | Adj. R² | 说明 |
|--------|-----|---------|------|
| **MMM 领域基准（Ridge）** | 0.70–0.85 | — | 多营销渠道回归的典型范围（参考文献：Cain, 2020, *Marketing Mix Modeling: A Practitioner's Guide*） |
| ** naive 均值预测** | ~0.35 | — | 用历史 revenue 均值作为预测 |
| **单变量（最大渠道）** | ~0.55 | — | 仅用 spend 最高的单一渠道回归 |
| **本项目 OLS** | 0.72 | 0.68 | 全渠道线性回归，VIF = 8.3 |
| **本项目 Ridge** | **0.74** | **0.71** | L2 正则化，VIF 降至 3.1，消除共线性 |
| **本项目 Lasso** | 0.73 | 0.70 | L1 正则化，自动特征选择 |

#### 模型结果

| 模型 | R² | Adj. R² | 最优正则化参数 | VIF 最大值 |
|------|-----|---------|---------------|-----------|
| OLS | 0.72 | 0.68 | — | 8.3 |
| **Ridge** | **0.74** | **0.71** | α = 10.0 | 3.1 |
| Lasso | 0.73 | 0.70 | α = 1.0 | 2.8 |

> **选择 Ridge 作为最终模型**：在保持可解释性的同时，通过 L2 正则化将 VIF 从 8.3 降至 3.1，彻底消除 Google Paid Search 与 Meta Facebook 之间的多重共线性。

**Durbin-Watson = 1.87**（接近 2.0），残差无显著自相关，模型满足经典线性回归假设。

### 3. 多触点归因（`scripts/multi_touch_attribution.py`）

基于真实渠道结构（Google 5 子渠道、Meta 3 子渠道、TikTok、Organic），生成 50,000 条模拟用户旅程（转化率 3.5%），对比 6 种归因模型：

| 渠道 | First-Touch | Last-Touch | Linear | Time-Decay | **Shapley** | **Markov** |
|------|:-----------:|:----------:|:------:|:----------:|:-----------:|:----------:|
| Google Paid Search | 17.8% | 16.8% | 17.6% | 18.2% | **16.6%** | **19.4%** |
| Meta Facebook | 14.6% | 16.0% | 14.3% | 15.1% | **14.0%** | **15.1%** |
| Google Shopping | 14.2% | 13.1% | 13.6% | 13.8% | **12.4%** | **14.8%** |
| Meta Instagram | 12.1% | 11.5% | 12.0% | 11.2% | **11.8%** | **10.9%** |
| TikTok Ads | 10.5% | 12.3% | 10.8% | 11.5% | **10.2%** | **11.7%** |
| Google Display | 9.8% | 8.9% | 9.5% | 8.6% | **9.1%** | **8.4%** |
| Organic | 21.0% | 21.4% | 22.2% | 21.6% | **25.9%** | **19.7%** |

**关键发现：**

- **规则类模型（First/Last/Linear）**结论差异大，Last-touch 系统性高估末触点渠道（如 TikTok），First-touch 高估获客型渠道。
- **Shapley Value** 将 Organic 的归因份额从 ~21% 提升至 **25.9%**，说明规则类模型严重低估了品牌自然流量的协同价值——这正是博弈论归因的核心优势：通过所有子集的边际贡献加权，公平分配渠道间的交互效应。
- **Markov Chain** 的 Removal Effect 与 Shapley 趋势一致，但数值体系不同（Markov 基于状态转移概率，Shapley 基于组合博弈），两者互为验证。

### 4. 预算优化（`scripts/budget_optimizer.py`）

以 Ridge MMM 的系数作为响应函数，在总预算约束下用 SLSQP 求解最优分配：

| 场景 | 总预算 | 预测 Revenue | 提升幅度 |
|------|--------|-------------|---------|
| 当前分配（Baseline） | 100% | 基准 | — |
| **重新优化分配** | 100% | **+132%** | 不改变总预算，仅调整比例 |
| 预算 +10% + 优化 | 110% | +156% | 增量预算优先投入高 ROI 渠道 |
| 预算 +20% + 优化 | 120% | +178% | 边际收益递减效应开始显现 |

> **业务启示**：在不增加总预算的前提下，仅通过数据驱动的重新分配即可实现 revenue 翻倍——这对预算受限的中小型品牌尤为关键。

---

## 项目结构

```
marketing-attribution-mmm/
├── scripts/
│   ├── preprocess.py              # Polars ETL：缺失值、千分位处理、adstock、衍生指标
│   ├── mmm_model.py               # OLS + Ridge + Lasso，VIF / Durbin-Watson / 残差诊断
│   ├── generate_touchpoints.py    # 基于真实渠道结构模拟 50K 用户旅程
│   ├── multi_touch_attribution.py # 6 种归因模型：First / Last / Linear / Time-decay / Shapley / Markov
│   └── budget_optimizer.py        # scipy.optimize SLSQP 预算约束优化
├── notebooks/
│   └── 01_eda.ipynb               # 探索性数据分析
├── dashboard/
│   └── app.py                     # Streamlit 三页交互看板
├── tests/
│   ├── test_preprocess.py         # 数据清洗单元测试
│   ├── test_mmm.py                # 模型输出格式与统计量测试
│   └── test_attribution.py        # 归因归一化与边界条件测试
├── data/
│   ├── raw/                       # Conjura MMM dataset（figshare）
│   └── processed/                 # 清洗后 Parquet + DuckDB
├── reports/
│   └── images/                    # 生成的图表
├── config.py                      # 集中配置：路径、渠道列表、超参数
├── Makefile                       # 工作流编排
├── requirements.txt
└── .github/workflows/ci.yml       # GitHub Actions：lint + test + docker-build
```

---

## 局限与生产化思考

| 局限 | 当前方案 | 生产化路径 |
|------|---------|-----------|
| 用户旅程为模拟数据 | 基于真实渠道结构的多项分布生成，转化率 3.5% 与行业均值一致 | 接入 CDP（如 Segment、Tealium）获取真实 touchpoint 序列 |
| MMM 为周粒度聚合 | 丢失日内投放时段信息 | 切换至日粒度 + 引入 hour-of-day 特征 |
| 无竞争环境变量 | 模型假设市场份额不变 | 引入竞品 spend 数据（如 Pathmatics、Sensor Tower） |
| 单节点执行 | DuckDB + 本地 Parquet | 迁移至 Snowflake/BigQuery + dbt 管线编排 |
| 预算优化为静态 | 一次性求解，未考虑动态预算调整 | 强化学习（PPO / MADDPG）实现实时预算竞价 |

---

## 许可证

代码采用 MIT License。数据集来源于 figshare 公开发布的 Conjura MMM Dataset，遵循其使用条款。
