# 全平台小说个性化推荐系统项目

## 项目简介
这是一个全平台小说个性化推荐系统，利用协同过滤等算法（如代码中 `model_training.py` 训练的模型 ），结合 Streamlit 搭建交互界面（`Universal_Novel_Recommendation_app.py` ），为用户推荐小说。

## 项目结构
- `data/`：存放小说相关数据（如用户评分、小说信息等，若有 ）
- `logos/`：存放平台图标等资源 
- `knn_model.pkl`、`svd_model.pkl`：训练好的推荐模型文件 
- `model_training.py`：模型训练脚本，用于生成推荐模型 
- `Universal_Novel_Recommendation_app.py`：Streamlit 应用主程序，实现交互和推荐功能 

## 环境依赖
需安装 Python 环境，依赖库可通过以下命令安装：
```bash
pip install pandas streamlit scikit-surprise 
