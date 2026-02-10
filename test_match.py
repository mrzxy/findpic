"""测试 photo.png 是否在 msg.png 中"""
from image_matcher import ImageMatcher
import cv2

# 创建匹配器
matcher = ImageMatcher(min_match_count=8, good_match_ratio=0.7)

print("正在分析图片...")
print("=" * 50)

# 执行匹配
result = matcher.find_image_in_image(
    small_img_path="photo.png",
    large_img_path="msg.png",
    visualize=True
)

print(f"\n【匹配结果】")
print(f"是否找到: {'✓ 是' if result['found'] else '✗ 否'}")
print(f"匹配点数: {result['match_count']}")
print(f"置信度: {result['confidence']:.2%}")
print(f"小图特征点数: {result['total_small_features']}")
print(f"大图特征点数: {result['total_large_features']}")

if result['found']:
    print(f"\n【详细信息】")
    if 'inliers' in result:
        print(f"内点数量: {result['inliers']}")

    if 'location' in result:
        location = result['location']
        print(f"\n【头像位置】")
        print(f"左上角: ({location[0][0]:.1f}, {location[0][1]:.1f})")
        print(f"左下角: ({location[1][0]:.1f}, {location[1][1]:.1f})")
        print(f"右下角: ({location[2][0]:.1f}, {location[2][1]:.1f})")
        print(f"右上角: ({location[3][0]:.1f}, {location[3][1]:.1f})")

        # 计算中心点
        center_x = sum(pt[0] for pt in location) / 4
        center_y = sum(pt[1] for pt in location) / 4
        print(f"中心点: ({center_x:.1f}, {center_y:.1f})")

    # 保存可视化结果
    if 'visualization' in result:
        cv2.imwrite('match_visualization.png', result['visualization'])
        print(f"\n✓ 匹配可视化已保存到: match_visualization.png")

    if 'marked_image' in result:
        cv2.imwrite('marked_chat.png', result['marked_image'])
        print(f"✓ 标记图像已保存到: marked_chat.png")

else:
    print(f"\n原因分析:")
    if result['match_count'] < 8:
        print(f"- 匹配点太少 ({result['match_count']} < 8)")
        print(f"- 可能的原因: 头像不在聊天记录中，或图片质量差异较大")

    if 'message' in result:
        print(f"- {result['message']}")

print("\n" + "=" * 50)
