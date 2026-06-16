# Contributing Guide

感谢你对本项目的兴趣. 本指南面向希望本地运行、调试或扩展该营销归因与预算优化项目的开发者.

## 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/MeaFew/attributor.git
cd attributor

# 2. 创建虚拟环境 (推荐 Python 3.12)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

## 数据准备

项目使用 figshare 公开发布的 Conjura MMM Dataset. 请运行下载脚本获取数据集:

```bash
bash download_data.sh
```

## 本地工作流

```bash
# 1. 数据预处理
make preprocess

# 2. MMM 建模 (OLS + Ridge + Lasso)
make mmm

# 3. 用户旅程模拟 + 多触点归因
make attribution

# 4. 预算优化
make optimize

# 5. 启动看板
make dashboard
```

## 代码规范

提交前请确保通过以下检查:

```bash
# Python lint
ruff check scripts/ dashboard/ --ignore E501,F401,E402

# 单元测试
pytest tests/ -v
```

## 提交规范

- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `refactor:` 重构
- `ci:` 持续集成相关
- `test:` 测试相关

## 扩展建议

- 新增归因模型: 放在 `scripts/` 并从 `multi_touch_attribution.py` 调用
- 新增分析脚本: 放在 `scripts/` 并按功能命名
- 新增 notebook: 放在 `notebooks/` 并更新 README 索引
