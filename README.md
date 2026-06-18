<p align="center">
  <h1 align="center">Marketing Attribution & Budget Optimization</h1>
  <p align="center">
    <b>从宏观 MMM 到微观多触点归因的全链路营销效果评估与预算优化系统</b>
  </p>
  <p align="center">
    <a href="https://github.com/MeaFew/attributor/actions"><img src="https://github.com/MeaFew/attributor/workflows/CI/badge.svg" alt="CI"></a>
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
- **归因模型选择无依据**：First-touch、Last-touch、Shapley Value、移除效应分析 (Removal Effect) 等方法结论差异巨大，如何系统比较？
- **预算分配凭经验**：在总预算约束下，如何科学重新分配各渠道 spend 以最大化 revenue？

---

## 技术架构

```mermaid
flowchart LR
    A[Raw CSV<br/>132K rows x 50 cols] --> B[Polars ETL]
    B --> C[Parquet]
    C --> D[MMM Modeling<br/>OLS / Ridge / Lasso]
    C --> E[User Journey<br/>Criteo 16.5M]
    E --> F[6 Attribution Models]
    D --> G[Budget Optimizer<br/>scipy SLSQP]
    F --> G
    G --> H[Streamlit Dashboard]
```

| 层级 | 技术选型 | 设计理由 |
|------|---------|---------|
| 数据清洗 | **Polars** | 向量化执行 + 惰性求值，处理 132K 行毫秒级 |
| 存储 | Parquet | 列式压缩，高效读写 |
| 宏观建模 | **statsmodels** + **scikit-learn** | OLS 提供完整统计推断（p-value、置信区间）；Ridge/Lasso 处理渠道间共线性 |
| 微观归因 | 自研 6 种模型 | 覆盖规则类（First/Last/Linear/Time-decay）与博弈论类（Shapley/移除效应分析），便于横向对比 |
| 预算优化 | **scipy.optimize** SLSQP | 支持等式约束（总预算不变）与不等式约束（单渠道下限），收敛稳定 |
| 交付 | **Streamlit** + **Plotly** | 三页交互看板：MMM 概览 / 归因对比 / 预算模拟器 |

---

## 快速开始

```bash
git clone https://github.com/MeaFew/attributor.git
cd attributor

# 1. 下载 MMM 数据集（GitHub Releases，约 31MB）
bash download_data.sh

# 2.（可选但推荐）下载真实归因数据集 Criteo Attribution Modeling for Bidding Dataset
#    约 623MB，下载后放到 data/raw/criteo_attribution_dataset.tsv.gz
#    官方：https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/

# 安装依赖并运行
make setup        # 创建虚拟环境 + 安装依赖
make all
# Windows (无 GNU Make): python run_all.py          # 运行完整管线：清洗 → MMM → 归因 → 优化
make dashboard    # 启动 Streamlit 交互看板
make verify       # 本地质量门（lint + format + test + audit）
```

---

## 核心模块

### 1. 数据预处理（`scripts/preprocess.py`）

```
输入: 132,759 rows x 50 cols（含大量空值与千分位逗号分隔符）
输出: 清洗后的 Parquet (日粒度)
关键操作:
  - 千分位逗号去除 + Float64 强制转换（解决 Polars 自动推断为 String 的问题）
  - CTR、CPM、ROAS 衍生指标计算
  - Adstock 衰减特征构造: x_t + 0.5*x_{t-1} + 0.25*x_{t-3} + 0.125*x_{t-7}
  - 时间特征提取（year/month/day_of_week/is_weekend）
```

### 2. 营销组合建模（`scripts/mmm_model.py`）

#### Benchmark

> Conjura MMM Dataset 为 2024 年 6 月发布的学术数据集，暂无官方竞赛 Baseline。以下为 MMM 领域公认的标准建模基准：

| 参照系 | R^2 | Adj. R^2 | 说明 |
|--------|-----|---------|------|
| **MMM 领域基准（Ridge）** | 0.70–0.85 | — | 品牌级、含完整促销/价格/竞争数据的 MMM（参考文献：Hanssens et al., *Market Response Models*, 2nd ed.） |
| **naive 均值预测** | ~0.0 | — | 用历史 revenue 均值作为预测 |
| **单变量（最大渠道）** | ~0.35 | — | 仅用 spend 最高的单一渠道回归 |
| **本项目 OLS** | 0.569 | 0.563 | 单品牌全渠道线性回归（auto-selected brand=5488b0） |
| **本项目 Ridge** | 0.569 | 0.563 | L2 正则化（alpha=1.0） |
| **本项目 Lasso** | 0.569 | 0.563 | L1 正则化（alpha=0.1） |

> R^2 ~ 0.57 反映了跨品牌聚合层面 MMM 的典型挑战：无价格/促销/竞品数据时，仅用渠道 spend 解释 revenue 变异的天然上限。品牌级细分模型可达 0.70–0.85。

#### 模型结果

| 模型 | R^2 | Adj. R^2 | 最优正则化参数 |
|------|-----|---------|---------------|
| OLS | 0.569 | 0.563 | — |
| **Ridge** | 0.569 | 0.563 | alpha = 1.0 |
| Lasso | 0.569 | 0.563 | alpha = 0.1 |

> Ridge/Lasso 在该数据上表现与 OLS 接近（spend 变量间共线性不高），正则化后系数更稳定。Durbin-Watson 统计量由模型自动输出，参见 `data/processed/models/mmm_results.json`。

### 3. 多触点归因（`scripts/multi_touch_attribution.py`）

使用真实 **Criteo Attribution Modeling for Bidding Dataset**（30 天实时流量，1,650 万条展示，61 万用户，4.5 万次转化）。将 impression 级数据按 `uid` 聚合成用户旅程，取 Top 10 campaigns 作为独立渠道，其余 665 个 campaign 归入 `other` 桶，对比 5 种归因模型 + 移除效应分析：

| 渠道 | First-Touch | Last-Touch | Linear | Time-Decay | **Shapley** | **Removal Eff.** |
|------|:-----------:|:----------:|:------:|:----------:|:-----------:|:----------:|
| campaign_10341182 | 4.2% | 5.1% | 4.2% | 5.0% | **4.2%** | **16.9%** |
| campaign_9100693 | 3.2% | 4.7% | 5.3% | 4.4% | **3.3%** | **19.4%** |
| campaign_15184511 | 3.3% | 3.2% | 2.7% | 3.2% | **3.2%** | **16.1%** |
| campaign_30801593 | 3.0% | 3.0% | 3.2% | 2.9% | **2.9%** | **1.1%** |
| campaign_32368244 | 2.4% | 2.9% | 2.4% | 2.8% | **2.4%** | **13.3%** |
| campaign_15398570 | 2.2% | 2.2% | 1.7% | 2.2% | **2.3%** | **5.8%** |
| campaign_29427842 | 1.6% | 1.7% | 1.6% | 1.7% | **1.9%** | **6.5%** |
| campaign_2869134 | 1.9% | 2.9% | 3.5% | 2.7% | **1.9%** | **12.0%** |
| campaign_31772643 | 1.2% | 1.1% | 0.8% | 1.1% | **1.3%** | **5.4%** |
| campaign_28351001 | 1.3% | 1.1% | 0.8% | 1.1% | **1.3%** | **3.5%** |
| **other** | **75.7%** | **72.0%** | **73.7%** | **73.0%** | **75.5%** | **0.0%** |

> 注：campaign ID 为 Criteo 数据中的匿名化广告活动 ID。`other` 聚合了除 Top 10 外的 665 个长尾 campaign，因此占据绝大部分展示与转化。

**关键发现：**

- **规则类模型（First/Last/Linear/Time-Decay）**结论高度一致：由于 `other` 桶覆盖了绝大多数展示，各规则模型都将其归因份额排在 72%–76% 之间。
- **Shapley Value** 在真实数据上仍提供了最稳定的分配，Top 3 渠道（campaign_10341182、campaign_9100693、campaign_15184511）合计贡献约 11%，与规则类模型趋势一致。
- **移除效应分析（Removal Effect）** 对 `other` 桶不敏感（移除 `other` 后剩余样本极少，导致其份额被压到 0%），但对头部单一 campaign 非常敏感——campaign_9100693 和 campaign_10341182 的移除效应分别达到 19.4% 和 16.9%，说明这两个 campaign 对整体转化率的边际影响最大。
- **方法启示**：真实数据下归因模型之间的差异比模拟数据更小（因为 `other` 桶占据主导），但 Shapley 与 Removal Effect 仍能有效识别头部高影响 campaign。

### 4. 预算优化（`scripts/budget_optimizer.py`）

以 Ridge MMM 的系数和截距作为线性响应函数，在总预算约束下用 SLSQP 求解最优分配：

| 场景 | 总预算 | 预测 Revenue | 提升幅度 |
|------|--------|-------------|---------|
| 当前分配（Baseline） | 100% | 基准 | — |
| **重新优化分配** | 100% | **+1.8%** | 不改变总预算，仅调整比例 |
| 预算 +10% + 优化 | 110% | +2.0% | 增量预算优先投入高 ROI 渠道 |
| 预算 +20% + 优化 | 120% | +2.1% | 边际收益递减效应开始显现 |

> **业务启示**：在不增加总预算的前提下，仅通过数据驱动的重新分配即可获得约 2% 的 revenue 提升。该框架可扩展至更丰富的特征（价格、促销、竞品）以提升优化空间。

---

## 项目结构

```
attributor/
├── scripts/
│   ├── preprocess.py              # Polars ETL：缺失值、千分位处理、adstock、衍生指标
│   ├── mmm_model.py               # OLS + Ridge + Lasso，VIF / Durbin-Watson / 残差诊断
│   ├── generate_touchpoints.py    # 基于真实渠道结构模拟 50K 用户旅程（fallback）
│   ├── preprocess_criteo.py       # 将 Criteo impression 数据聚合为用户旅程
│   ├── multi_touch_attribution.py # 6 种归因模型：First / Last / Linear / Time-decay / Shapley / Removal Effect
│   └── budget_optimizer.py        # scipy.optimize SLSQP 预算约束优化
├── notebooks/
│   └── 01_eda.ipynb               # 探索性数据分析
├── dashboard/
│   └── app.py                     # Streamlit 三页交互看板
├── tests/
│   ├── test_algorithms.py         # 算法单元测试：6 归因模型（含 Shapley 数学正确性）+ MMM 三模型 + 预算优化器
│   ├── test_preprocess.py         # 数据清洗单元测试
│   ├── test_mmm.py                # 模型输出格式与统计量测试
│   └── test_attribution.py        # 归因归一化与边界条件测试
├── data/
│   ├── raw/                       # Conjura MMM dataset（figshare）
│   └── processed/                 # 清洗后 Parquet
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
| 多触点归因使用 Criteo 真实数据，但 campaign 已聚合 | 取 Top 10 campaigns + `other` 桶（665 个 campaign），便于 Shapley 计算 | 直接接入 CDP/Segment/Tealium 获取完整、未聚合的用户旅程；或使用更多计算资源处理全部 campaign |
| MMM 为日粒度 | 原始数据的日粒度已提供一定的时间分辨率  | 引入 hour-of-day 或 daypart 特征进一步精细化 |
| 无竞争环境变量 | 模型假设市场份额不变 | 引入竞品 spend 数据（如 Pathmatics、Sensor Tower） |
| 单节点执行 | 本地 Parquet | 迁移至 Snowflake/BigQuery + dbt 管线编排 |
| 预算优化为静态 | 一次性求解，未考虑动态预算调整 | 强化学习（PPO / MADDPG）实现实时预算竞价 |

---

## 相关项目

| 项目 | 仓库 | 简介 |
|------|------|------|
| 电商用户行为分析 | [MeaFew/shoplytics](https://github.com/MeaFew/shoplytics) | 2,900万条真实用户行为数据，10大分析模块 |
| 信用风险评分 | [MeaFew/riskscore](https://github.com/MeaFew/riskscore) | WOE/IV + XGBoost/LightGBM + SHAP 可解释性 |
| 多元时序预测 | [MeaFew/foresight](https://github.com/MeaFew/foresight) | LSTM / Transformer / XGBoost 时序预测对比 |

| 图神经网络反欺诈 | [MeaFew/graphguard](https://github.com/MeaFew/graphguard) | 图神经网络非法交易检测 |
## 许可证

代码采用 MIT License。数据集来源于 figshare 公开发布的 Conjura MMM Dataset，遵循其使用条款。
