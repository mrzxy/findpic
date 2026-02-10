"""
示例：如何使用 ImageMatcher 判断用户头像是否在聊天记录中
"""
from image_matcher import ImageMatcher
import cv2


def example_simple_match():
    """示例1：简单匹配"""
    print("=== 示例1：简单匹配 ===")

    matcher = ImageMatcher(min_match_count=10)

    result = matcher.find_image_in_image(
        small_img_path="avatar.png",
        large_img_path="chat_screenshot.png",
        visualize=True
    )

    if result['found']:
        print(f"✓ 找到用户头像！")
        print(f"  - 匹配点数: {result['match_count']}")
        print(f"  - 置信度: {result['confidence']:.2%}")

        # 保存结果
        cv2.imwrite('result.png', result['visualization'])
    else:
        print(f"✗ 未找到用户头像")
        print(f"  - 匹配点数: {result['match_count']} (需要至少 10 个)")


def example_batch_match():
    """示例2：批量匹配多张聊天记录"""
    print("\n=== 示例2：批量匹配 ===")

    matcher = ImageMatcher(min_match_count=8)

    chat_screenshots = [
        "chat1.png",
        "chat2.png",
        "chat3.png"
    ]

    results = matcher.batch_find("avatar.png", chat_screenshots)

    for result in results:
        status = "✓ 找到" if result['found'] else "✗ 未找到"
        print(f"{status} - {result['image_path']} (匹配点: {result['match_count']})")


def example_custom_params():
    """示例3：自定义参数以适应不同场景"""
    print("\n=== 示例3：自定义参数 ===")

    # 对于高相似度要求：提高最小匹配点数
    strict_matcher = ImageMatcher(min_match_count=15, good_match_ratio=0.65)

    # 对于模糊或部分遮挡的情况：降低要求
    relaxed_matcher = ImageMatcher(min_match_count=6, good_match_ratio=0.75)

    result = relaxed_matcher.find_image_in_image(
        "avatar.png",
        "blurry_chat.png"
    )

    print(f"宽松匹配结果: {'找到' if result['found'] else '未找到'}")


def example_get_location():
    """示例4：获取头像在聊天记录中的位置"""
    print("\n=== 示例4：获取位置信息 ===")

    matcher = ImageMatcher()
    result = matcher.find_image_in_image(
        "avatar.png",
        "chat_screenshot.png",
        visualize=True
    )

    if result['found'] and 'location' in result:
        # 位置是四个角点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        location = result['location']
        print(f"找到头像，位置坐标:")
        print(f"  左上: ({location[0][0]:.1f}, {location[0][1]:.1f})")
        print(f"  左下: ({location[1][0]:.1f}, {location[1][1]:.1f})")
        print(f"  右下: ({location[2][0]:.1f}, {location[2][1]:.1f})")
        print(f"  右上: ({location[3][0]:.1f}, {location[3][1]:.1f})")

        # 计算中心点
        center_x = sum(pt[0] for pt in location) / 4
        center_y = sum(pt[1] for pt in location) / 4
        print(f"  中心: ({center_x:.1f}, {center_y:.1f})")

        # 保存标记了位置的图像
        if 'marked_image' in result:
            cv2.imwrite('avatar_location.png', result['marked_image'])
            print("标记图像已保存到 avatar_location.png")


def example_filter_messages():
    """示例5：实际应用 - 过滤出特定用户的消息"""
    print("\n=== 示例5：实际应用场景 ===")

    matcher = ImageMatcher(min_match_count=8)

    # 假设你有多张聊天截图
    chat_images = ["chat_page1.png", "chat_page2.png", "chat_page3.png"]
    target_avatar = "target_user_avatar.png"

    user_messages = []

    for chat_img in chat_images:
        result = matcher.find_image_in_image(target_avatar, chat_img, visualize=False)

        if result['found']:
            user_messages.append({
                'screenshot': chat_img,
                'confidence': result['confidence'],
                'location': result.get('location')
            })

    print(f"在 {len(chat_images)} 张聊天记录中找到 {len(user_messages)} 张包含目标用户的截图")
    for msg in user_messages:
        print(f"  - {msg['screenshot']} (置信度: {msg['confidence']:.2%})")


if __name__ == "__main__":
    # 运行所有示例（需要准备相应的图片文件）
    # example_simple_match()
    # example_batch_match()
    # example_custom_params()
    # example_get_location()
    # example_filter_messages()

    print("请根据需要取消注释相应的示例函数")
    print("\n使用前请准备：")
    print("1. 用户头像图片（如 avatar.png）")
    print("2. 聊天记录截图（如 chat_screenshot.png）")
