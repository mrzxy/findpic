"""
智能头像匹配器
处理圆形裁剪、颜色差异等问题
"""
import cv2
import numpy as np


class SmartAvatarMatcher:
    """智能头像匹配器，适配各种头像变换"""

    def __init__(self):
        pass

    def preprocess_for_matching(self, img, apply_mask=True):
        """
        预处理图像以提高匹配率

        Args:
            img: 输入图像
            apply_mask: 是否应用圆形遮罩

        Returns:
            处理后的图像
        """
        # 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 如果需要应用圆形遮罩（模拟圆形头像）
        if apply_mask:
            h, w = gray.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            radius = min(w, h) // 2 - 2
            cv2.circle(mask, center, radius, 255, -1)
            gray = cv2.bitwise_and(gray, gray, mask=mask)

        return gray

    def find_avatar(self, avatar_path, chat_path, visualize=False):
        """
        查找头像在聊天截图中的位置

        Args:
            avatar_path: 头像图片路径
            chat_path: 聊天截图路径
            visualize: 是否生成可视化

        Returns:
            dict: 匹配结果
        """
        # 读取图像
        avatar = cv2.imread(avatar_path)
        chat = cv2.imread(chat_path)

        if avatar is None or chat is None:
            raise ValueError("无法读取图像")

        # 多种匹配策略
        results = []

        # 策略1: 直接模板匹配（多尺度）
        result1 = self._match_template(avatar, chat, apply_mask=False)
        results.append(('直接匹配', result1))

        # 策略2: 圆形遮罩模板匹配
        result2 = self._match_template(avatar, chat, apply_mask=True)
        results.append(('圆形遮罩', result2))

        # 策略3: 只匹配中心区域（避免边缘）
        result3 = self._match_center_region(avatar, chat)
        results.append(('中心区域', result3))

        # 策略4: 边缘检测匹配
        result4 = self._match_edges(avatar, chat)
        results.append(('边缘匹配', result4))

        # 选择最佳结果
        best_strategy, best_result = max(results, key=lambda x: x[1]['score'])

        # 生成可视化
        if visualize and best_result['location'] is not None:
            img_vis = chat.copy()
            x, y, w, h = best_result['location']

            # 绘制矩形
            color = (0, 255, 0) if best_result['found'] else (0, 165, 255)
            cv2.rectangle(img_vis, (x, y), (x + w, y + h), color, 2)

            # 绘制圆形（模拟圆形头像）
            center_x, center_y = x + w // 2, y + h // 2
            radius = min(w, h) // 2
            cv2.circle(img_vis, (center_x, center_y), radius, color, 2)

            # 添加文本
            text = f"{best_strategy}: {best_result['score']:.2%}"
            cv2.putText(img_vis, text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            best_result['visualization'] = img_vis

        best_result['strategy'] = best_strategy
        return best_result

    def _match_template(self, avatar, chat, apply_mask=False):
        """模板匹配"""
        # 预处理
        avatar_processed = self.preprocess_for_matching(avatar, apply_mask)
        chat_gray = cv2.cvtColor(chat, cv2.COLOR_BGR2GRAY)

        best_score = 0
        best_loc = None
        best_scale = 1.0

        # 多尺度匹配
        for scale in np.arange(0.3, 1.5, 0.1):
            h, w = avatar_processed.shape
            new_h, new_w = int(h * scale), int(w * scale)

            if new_h < 10 or new_w < 10 or new_h > chat_gray.shape[0] or new_w > chat_gray.shape[1]:
                continue

            resized = cv2.resize(avatar_processed, (new_w, new_h))

            # 模板匹配
            result = cv2.matchTemplate(chat_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_loc = max_loc
                best_scale = scale

        if best_loc is not None:
            h, w = avatar_processed.shape
            scaled_h, scaled_w = int(h * best_scale), int(w * best_scale)
            return {
                'found': best_score > 0.3,
                'score': best_score,
                'location': (best_loc[0], best_loc[1], scaled_w, scaled_h),
                'scale': best_scale
            }

        return {'found': False, 'score': 0, 'location': None, 'scale': 1.0}

    def _match_center_region(self, avatar, chat):
        """只匹配头像中心区域（80%）"""
        h, w = avatar.shape[:2]
        crop_size = int(min(h, w) * 0.8)
        start_y = (h - crop_size) // 2
        start_x = (w - crop_size) // 2

        avatar_center = avatar[start_y:start_y+crop_size, start_x:start_x+crop_size]
        return self._match_template(avatar_center, chat, apply_mask=False)

    def _match_edges(self, avatar, chat):
        """基于边缘检测的匹配"""
        # 边缘检测
        avatar_gray = cv2.cvtColor(avatar, cv2.COLOR_BGR2GRAY)
        chat_gray = cv2.cvtColor(chat, cv2.COLOR_BGR2GRAY)

        avatar_edges = cv2.Canny(avatar_gray, 50, 150)
        chat_edges = cv2.Canny(chat_gray, 50, 150)

        best_score = 0
        best_loc = None
        best_scale = 1.0

        for scale in np.arange(0.3, 1.5, 0.1):
            h, w = avatar_edges.shape
            new_h, new_w = int(h * scale), int(w * scale)

            if new_h < 10 or new_w < 10 or new_h > chat_edges.shape[0] or new_w > chat_edges.shape[1]:
                continue

            resized = cv2.resize(avatar_edges, (new_w, new_h))
            result = cv2.matchTemplate(chat_edges, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_loc = max_loc
                best_scale = scale

        if best_loc is not None:
            h, w = avatar_edges.shape
            scaled_h, scaled_w = int(h * best_scale), int(w * best_scale)
            return {
                'found': best_score > 0.25,
                'score': best_score,
                'location': (best_loc[0], best_loc[1], scaled_w, scaled_h),
                'scale': best_scale
            }

        return {'found': False, 'score': 0, 'location': None, 'scale': 1.0}


if __name__ == "__main__":
    matcher = SmartAvatarMatcher()

    result = matcher.find_avatar(
        avatar_path="photo.png",
        chat_path="msg.png",
        visualize=True
    )

    print(f"匹配策略: {result['strategy']}")
    print(f"找到: {result['found']}")
    print(f"分数: {result['score']:.2%}")

    if result['location']:
        x, y, w, h = result['location']
        print(f"位置: ({x}, {y}), 大小: {w}x{h}")

    if 'visualization' in result:
        cv2.imwrite('smart_match_result.png', result['visualization'])
        print("已保存: smart_match_result.png")
