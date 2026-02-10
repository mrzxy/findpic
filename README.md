# 图像匹配工具 - 头像识别

判断用户头像是否存在于聊天记录截图中，支持多种匹配算法。

## 功能特点

- ✅ **智能匹配**: 自动选择最佳匹配策略（SIFT、模板匹配、边缘检测等）
- ✅ **圆形头像支持**: 专门优化处理聊天软件中的圆形头像
- ✅ **多尺度匹配**: 自动处理大小变化（0.3x - 1.5x）
- ✅ **精确定位**: 返回头像在截图中的坐标和大小
- ✅ **置信度评分**: 提供匹配可靠性分数
- ✅ **可视化结果**: 自动生成标注图像

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 🚀 命令行使用（推荐）

```bash
python find_avatar.py photo.png msg.png
```

输出示例：
```
✅ 找到匹配！

策略: 中心区域
置信度: 48.99%

位置信息:
  坐标: (54, 32)
  大小: 30 × 30 像素
  中心: (69, 47)

✓ 结果图已保存: avatar_found.png
```

### 💻 Python 代码使用

#### 方法1: 智能匹配器（推荐，适合聊天头像）

```python
from smart_avatar_matcher import SmartAvatarMatcher

matcher = SmartAvatarMatcher()

result = matcher.find_avatar(
    avatar_path="photo.png",
    chat_path="msg.png",
    visualize=True
)

if result['found']:
    print(f"✓ 找到！置信度: {result['score']:.2%}")
    x, y, w, h = result['location']
    print(f"位置: ({x}, {y}), 大小: {w}×{h}")
```

#### 方法2: SIFT 特征匹配（适合大图、清晰图片）

```python
from image_matcher import ImageMatcher

matcher = ImageMatcher(min_match_count=10)

result = matcher.find_image_in_image(
    small_img_path="avatar.png",
    large_img_path="chat.png",
    visualize=True
)

if result['found']:
    print(f"找到匹配！置信度: {result['confidence']:.2%}")
```

### 返回结果说明

```python
{
    'found': True,              # 是否找到匹配
    'match_count': 25,          # 匹配的特征点数量
    'confidence': 0.85,         # 置信度 (0-1)
    'inliers': 23,             # RANSAC 内点数量
    'location': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # 四个角点坐标
    'total_small_features': 150,  # 小图特征点总数
    'total_large_features': 1200, # 大图特征点总数
    'visualization': <图像>,     # 匹配可视化 (如果 visualize=True)
    'marked_image': <图像>       # 标记了匹配区域的大图
}
```

## 使用场景

### 场景1：判断消息是否由特定用户发送

```python
matcher = ImageMatcher()

# 检查聊天记录中是否有该用户的消息
result = matcher.find_image_in_image("user_avatar.png", "chat_screenshot.png")

if result['found']:
    print("这条消息是该用户发送的")
```

### 场景2：批量筛选包含特定用户的聊天记录

```python
matcher = ImageMatcher(min_match_count=8)

chat_files = ["chat1.png", "chat2.png", "chat3.png"]
results = matcher.batch_find("target_avatar.png", chat_files)

# 筛选出包含目标用户的截图
user_chats = [r for r in results if r['found']]
print(f"找到 {len(user_chats)} 张包含目标用户的聊天记录")
```

### 场景3：获取头像在截图中的位置

```python
result = matcher.find_image_in_image("avatar.png", "chat.png", visualize=True)

if result['found']:
    location = result['location']
    # 计算中心点
    center_x = sum(pt[0] for pt in location) / 4
    center_y = sum(pt[1] for pt in location) / 4
    print(f"头像位于 ({center_x:.0f}, {center_y:.0f})")

    # 保存标记了位置的图像
    cv2.imwrite('result.png', result['marked_image'])
```

## 参数调优

### min_match_count (最小匹配点数)

- **默认**: 10
- **说明**: 判定为匹配所需的最小特征点数量
- **调优建议**:
  - 头像清晰、大图高清: 可提高到 15-20
  - 头像模糊或很小: 降低到 6-8
  - 头像可能被部分遮挡: 降低到 5-7

```python
# 严格模式
strict_matcher = ImageMatcher(min_match_count=15)

# 宽松模式（适合低质量图片）
relaxed_matcher = ImageMatcher(min_match_count=6)
```

### good_match_ratio (好匹配比率)

- **默认**: 0.7
- **说明**: Lowe's ratio test 阈值，用于过滤匹配质量
- **调优建议**:
  - 需要更高精度: 降低到 0.6-0.65
  - 图像质量差或有形变: 提高到 0.75-0.8

```python
# 高精度匹配
high_precision = ImageMatcher(good_match_ratio=0.65)

# 容错匹配
tolerant = ImageMatcher(good_match_ratio=0.75)
```

## 常见问题

### Q: 为什么找不到明明存在的头像？

A: 可能的原因：
1. 头像太小（特征点不足）- 尝试使用更大的头像图片
2. 图片质量差 - 降低 `min_match_count`
3. 头像被大幅缩放或旋转 - SIFT 应该能处理，但可以尝试调整参数
4. 头像和截图来源不同（不是同一个头像）

### Q: 如何提高匹配速度？

A:
1. 不需要可视化时设置 `visualize=False`
2. 预先裁剪大图，只保留聊天消息区域
3. 适当提高 `min_match_count`，减少误匹配的计算

### Q: 误匹配率高怎么办？

A:
1. 提高 `min_match_count` 到 15 或更高
2. 降低 `good_match_ratio` 到 0.6-0.65
3. 检查返回的 `confidence` 值，设置阈值过滤

## 技术原理

1. **SIFT 特征提取**: 检测图像中的关键点和特征描述符
2. **FLANN 匹配**: 使用快速近似最近邻搜索匹配特征点
3. **Lowe's Ratio Test**: 过滤低质量匹配
4. **RANSAC**: 计算单应性矩阵，去除异常匹配点
5. **位置计算**: 通过单应性矩阵变换得到精确位置

## 依赖库

- OpenCV (opencv-python 和 opencv-contrib-python)
- NumPy

## 许可

MIT License
# findpic
