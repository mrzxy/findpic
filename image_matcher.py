import cv2
import numpy as np


class ImageMatcher:
    """基于 SIFT 的图像匹配器，用于判断小图是否存在于大图中"""

    def __init__(self, min_match_count=10, good_match_ratio=0.7):
        """
        初始化图像匹配器

        Args:
            min_match_count: 判定为匹配的最小特征点数量
            good_match_ratio: Lowe's ratio test 的阈值
        """
        self.min_match_count = min_match_count
        self.good_match_ratio = good_match_ratio
        # 创建 SIFT 检测器
        self.sift = cv2.SIFT_create()
        # 创建 FLANN 匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def find_image_in_image(self, small_img_path, large_img_path, visualize=False):
        """
        判断小图是否存在于大图中

        Args:
            small_img_path: 小图路径（用户头像）
            large_img_path: 大图路径（聊天记录截图）
            visualize: 是否生成可视化结果图像

        Returns:
            dict: 包含匹配结果的字典
                - found: bool, 是否找到匹配
                - match_count: int, 匹配的特征点数量
                - confidence: float, 匹配置信度 (0-1)
                - location: tuple, 匹配区域的四个角点坐标 (如果找到)
                - visualization: np.ndarray, 可视化图像 (如果 visualize=True)
        """
        # 读取图像
        img_small = cv2.imread(small_img_path, cv2.IMREAD_GRAYSCALE)
        img_large = cv2.imread(large_img_path, cv2.IMREAD_GRAYSCALE)

        if img_small is None:
            raise ValueError(f"无法读取小图: {small_img_path}")
        if img_large is None:
            raise ValueError(f"无法读取大图: {large_img_path}")

        # 提取特征点和描述符
        kp_small, des_small = self.sift.detectAndCompute(img_small, None)
        kp_large, des_large = self.sift.detectAndCompute(img_large, None)

        # 检查是否提取到特征点
        if des_small is None or len(des_small) < 2:
            return {
                'found': False,
                'match_count': 0,
                'confidence': 0.0,
                'message': '小图特征点不足'
            }

        if des_large is None or len(des_large) < 2:
            return {
                'found': False,
                'match_count': 0,
                'confidence': 0.0,
                'message': '大图特征点不足'
            }

        # 使用 FLANN 进行特征匹配
        matches = self.flann.knnMatch(des_small, des_large, k=2)

        # 使用 Lowe's ratio test 筛选好的匹配
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.good_match_ratio * n.distance:
                    good_matches.append(m)

        match_count = len(good_matches)
        confidence = min(match_count / self.min_match_count, 1.0)

        result = {
            'found': match_count >= self.min_match_count,
            'match_count': match_count,
            'confidence': confidence,
            'total_small_features': len(kp_small),
            'total_large_features': len(kp_large)
        }

        # 如果找到足够的匹配点，计算匹配区域
        if result['found']:
            # 获取匹配点的坐标
            src_pts = np.float32([kp_small[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_large[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 使用 RANSAC 算法找到单应性矩阵
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                # 获取小图的四个角点
                h, w = img_small.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

                # 将角点转换到大图坐标系
                dst = cv2.perspectiveTransform(pts, M)
                result['location'] = dst.reshape(-1, 2).tolist()

                # 计算内点比例，更新置信度
                inliers = np.sum(mask)
                result['inliers'] = int(inliers)
                result['confidence'] = min((inliers / self.min_match_count) * 0.8 + 0.2, 1.0)

        # 生成可视化结果
        if visualize:
            img_small_color = cv2.imread(small_img_path)
            img_large_color = cv2.imread(large_img_path)

            if result['found'] and 'location' in result:
                # 在大图上绘制匹配区域
                pts = np.int32(result['location']).reshape(-1, 1, 2)
                img_result = cv2.polylines(img_large_color, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)

                # 绘制匹配的特征点连线
                draw_params = dict(
                    matchColor=(0, 255, 0),
                    singlePointColor=None,
                    matchesMask=mask.ravel().tolist() if 'inliers' in result else None,
                    flags=2
                )
                img_matches = cv2.drawMatches(img_small_color, kp_small, img_large_color, kp_large,
                                             good_matches, None, **draw_params)
                result['visualization'] = img_matches
                result['marked_image'] = img_result
            else:
                # 只绘制特征点匹配
                img_matches = cv2.drawMatches(img_small_color, kp_small, img_large_color, kp_large,
                                             good_matches[:20], None, flags=2)
                result['visualization'] = img_matches

        return result

    def batch_find(self, small_img_path, large_img_paths):
        """
        在多张大图中查找小图

        Args:
            small_img_path: 小图路径
            large_img_paths: 大图路径列表

        Returns:
            list: 每张大图的匹配结果列表
        """
        results = []
        for large_img_path in large_img_paths:
            result = self.find_image_in_image(small_img_path, large_img_path)
            result['image_path'] = large_img_path
            results.append(result)
        return results


def main():
    """示例用法"""
    # 创建匹配器实例
    matcher = ImageMatcher(min_match_count=10, good_match_ratio=0.7)

    # 执行匹配
    small_img = "avatar.png"  # 用户头像
    large_img = "chat_screenshot.png"  # 聊天记录截图

    try:
        result = matcher.find_image_in_image(small_img, large_img, visualize=True)

        print(f"匹配结果: {'找到' if result['found'] else '未找到'}")
        print(f"匹配点数量: {result['match_count']}")
        print(f"置信度: {result['confidence']:.2%}")
        print(f"小图特征点总数: {result['total_small_features']}")
        print(f"大图特征点总数: {result['total_large_features']}")

        if result['found']:
            print(f"内点数量: {result.get('inliers', 'N/A')}")
            print(f"匹配位置: {result.get('location', 'N/A')}")

            # 保存可视化结果
            if 'visualization' in result:
                cv2.imwrite('match_result.png', result['visualization'])
                print("匹配可视化已保存到 match_result.png")

            if 'marked_image' in result:
                cv2.imwrite('marked_result.png', result['marked_image'])
                print("标记图像已保存到 marked_result.png")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
