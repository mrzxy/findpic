"""使用模板匹配测试 photo.png 是否在 msg.png 中"""
from template_matcher import TemplateMatcher
import cv2

print("=" * 60)
print("使用模板匹配算法")
print("=" * 60)

# 创建模板匹配器（降低阈值，因为可能有颜色差异）
matcher = TemplateMatcher(threshold=0.4, method='TM_CCOEFF_NORMED')

print("\n正在分析...")

result = matcher.find_template(
    template_path="photo.png",
    image_path="msg.png",
    visualize=True,
    multi_scale=True  # 尝试不同大小
)

print(f"\n【匹配结果】")
print(f"是否找到: {'✓ 是' if result['found'] else '✗ 否'}")
print(f"匹配分数: {result['score']:.4f} ({result['confidence']:.2%})")
print(f"使用缩放: {result['scale']:.2f}x")

if 'location' in result:
    loc = result['location']
    print(f"\n【位置信息】")
    print(f"左上角坐标: ({loc['x']}, {loc['y']})")
    print(f"大小: {loc['width']} x {loc['height']} 像素")
    print(f"中心点: ({loc['center_x']}, {loc['center_y']})")

    # 保存结果
    if 'visualization' in result:
        cv2.imwrite('template_match_result.png', result['visualization'])
        print(f"\n✓ 可视化结果已保存: template_match_result.png")

if not result['found']:
    print(f"\n说明: 匹配分数 {result['score']:.2%} 低于阈值 {matcher.threshold:.2%}")
    print("可能原因:")
    print("- 头像被圆形裁剪，形状不同")
    print("- 颜色或亮度有差异")
    print("- 大小变化较大")
    print("\n建议: 查看 template_match_result.png 确认最佳匹配位置")

print("\n" + "=" * 60)
