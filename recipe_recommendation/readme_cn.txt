=============================
菜谱推荐系统（Recipe Recommendation）
=============================

本项目实现了一个完整的菜谱推荐系统，包括：
- 冷启动（Cold Start）排序  
- 机器学习模型（ML）重排序  
- KMeans 聚类多样化  
- 用户反馈收集与自动重训

所有功能都已封装好，外部调用只需要几个简单的接口。

----------------------------------------
1. 外部主要调用函数
----------------------------------------

1) recommend_recipes(detection_payload, user_id, recipes_df, topk=5)
   - 输入：
     • detection_payload：dict 或 JSON，表示检测到的食材  
     • user_id：str，用户 ID  
     • recipes_df：通过 `load_recipes()` 加载的菜谱 DataFrame  
     • topk：返回的推荐菜谱数量（默认 5）
   - 输出：
     • ml_top：推荐结果（DataFrame）  
     • user_parents：映射后的父食材列表  
     • high_conf：高置信度匹配  
     • low_conf：低置信度/未匹配食材

   功能包括：食材映射 → 相似用户模型复制 → 冷启动 → 粗排 → ML 重排 → KMeans 多样化。

2) load_recipes()
   - 自动从 Hugging Face（iris314）下载菜谱数据到 `data/`，并返回 DataFrame。

3) get_feedback(user_id, recipe_row, qid=None)
   - 收集用户反馈并写入 `user_data/{user_id}/feedback.csv`  
   - user_profile 自动加载，qid 缺省自动分配

----------------------------------------
2. 用户数据
----------------------------------------

`user_data` 里包含四个示例用户：
- user_0：空 profile，用于测试零信息自启
- user_1 / user_2：有不同饮食偏好的真实用户
- user_3：与 user_2 类似，用于测试 embedding 复制功能

每个用户目录下都有 `user_profile.json`、`user_features_rank.csv`、`ranker.pkl`。  
你可以新增用户，只需遵循以下 JSON 格式：

{
  "user_id": "user_001",
  "num_feedback": 0,
  "diet": {"vegetarian_type": "flexible_vegetarian"},
  "allergies": ["peanut", "shrimp"],
  "region_preference": ["Asia", "Europe"],
  "nutritional_goals": {
    "calories": {"min": 400, "max": 3000},
    "protein": {"min": 100, "max": 160}
  },
  "other_preferences": {
    "preferred_main": ["chicken", "tofu"],
    "disliked_main": ["lamb"],
    "cooking_time_max": 40
  }
}

冷启动过程通常需要 15～25 分钟（视机器性能而定）。

----------------------------------------
3. 数据下载
----------------------------------------

菜谱和食材映射等大文件会自动从 Hugging Face（iris314）下载并缓存到 `data/`，无需手动设置。

----------------------------------------
4. 快速上手示例
----------------------------------------

```python
from main import recommend_recipes, load_recipes, get_feedback

# 加载菜谱
recipes_df = load_recipes()

# 准备模拟检测输入
payload = {"detected_ingredients": ["chicken", "milk", "flour"]}

# 获取推荐结果
top_recipes, user_parents, high_conf, low_conf = recommend_recipes(payload, "user_1", recipes_df, topk=5)

# 提交反馈
get_feedback("user_1", top_recipes.iloc[0].to_dict())
