"""
基于模板匹配的图像查找器
更适合小图像、头像等场景
"""
import cv2
import numpy as np


class TemplateMatcher:
    """模板匹配器，适合查找头像等小图像"""

    def __init__(self, threshold=0.6, method='TM_CCOEFF_NORMED'):
        """
        初始化模板匹配器

        Args:
            threshold: 匹配阈值 (0-1)，越高越严格
            method: 匹配方法，可选：
                - 'TM_CCOEFF_NORMED' (推荐，归一化相关系数)
                - 'TM_CCORR_NORMED' (归一化相关)
                - 'TM_SQDIFF_NORMED' (归一化平方差，越小越好)
        """
        self.threshold = threshold
        self.method = getattr(cv2, method)
        self.method_name = method

    def find_template(self, template_path, image_path, visualize=False, multi_scale=True):
        """
        在大图中查找模板

        Args:
            template_path: 模板图片路径（小图/头像）
            image_path: 目标图片路径（大图/聊天记录）
            visualize: 是否生成可视化结果
            multi_scale: 是否使用多尺度匹配（处理大小变化）

        Returns:
            dict: 匹配结果
        """
        # 读取图像
        template = cv2.imread(template_path)
        image = cv2.imread(image_path)

        if template is None:
            raise ValueError(f"无法读取模板图片: {template_path}")
        if image is None:
            raise ValueError(f"无法读取目标图片: {image_path}")

        # 转换为灰度图
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        best_match = None
        best_score = -1 if 'SQDIFF' not in self.method_name else float('inf')

        # 多尺度匹配
        if multi_scale:
            scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        else:
            scales = [1.0]

        for scale in scales:
            # 调整模板大小
            if scale != 1.0:
                width = int(template_gray.shape[1] * scale)
                height = int(template_gray.shape[0] * scale)
                if width < 5 or height < 5:
                    continue
                scaled_template = cv2.resize(template_gray, (width, height))
            else:
                scaled_template = template_gray

            # 检查模板是否大于图像
            if (scaled_template.shape[0] > image_gray.shape[0] or
                scaled_template.shape[1] > image_gray.shape[1]):
                continue

            # 执行模板匹配
            result = cv2.matchTemplate(image_gray, scaled_template, self.method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # 根据方法选择最佳值
            if 'SQDIFF' in self.method_name:
                score = 1 - min_val  # 转换为越大越好
                loc = min_loc
                current_score = min_val
            else:
                score = max_val
                loc = max_loc
                current_score = max_val

            # 更新最佳匹配
            if 'SQDIFF' in self.method_name:
                if current_score < best_score:
                    best_score = current_score
                    best_match = {
                        'location': loc,
                        'scale': scale,
                        'template_size': (scaled_template.shape[1], scaled_template.shape[0]),
                        'score': score
                    }
            else:
                if current_score > best_score:
                    best_score = current_score
                    best_match = {
                        'location': loc,
                        'scale': scale,
                        'template_size': (scaled_template.shape[1], scaled_template.shape[0]),
                        'score': score
                    }

        # 判断是否找到匹配
        if best_match is None:
            return {
                'found': False,
                'confidence': 0.0,
                'message': '无法进行匹配'
            }

        found = best_match['score'] >= self.threshold
        x, y = best_match['location']
        w, h = best_match['template_size']

        result_dict = {
            'found': found,
            'confidence': best_match['score'],
            'score': best_match['score'],
            'scale': best_match['scale'],
            'location': {
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'center_x': x + w // 2,
                'center_y': y + h // 2
            },
            'bbox': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        }

        # 生成可视化
        if visualize:
            img_result = image.copy()

            if found:
                # 绘制矩形框
                cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 添加文本
                text = f"Match: {best_match['score']:.2%}"
                cv2.putText(img_result, text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 绘制中心点
                center_x, center_y = x + w // 2, y + h // 2
                cv2.circle(img_result, (center_x, center_y), 5, (0, 0, 255), -1)
            else:
                # 即使未找到，也标记最佳位置（用红色）
                cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                text = f"Best: {best_match['score']:.2%} (< {self.threshold:.2%})"
                cv2.putText(img_result, text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            result_dict['visualization'] = img_result

        return result_dict


def main():
    """测试模板匹配"""
    matcher = TemplateMatcher(threshold=0.5)

    result = matcher.find_template(
        template_path="photo.png",
        image_path="msg.png",
        visualize=True,
        multi_scale=True
    )

    print(f"匹配结果: {'找到' if result['found'] else '未找到'}")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"缩放比例: {result['scale']:.2f}")

    if 'location' in result:
        loc = result['location']
        print(f"位置: ({loc['x']}, {loc['y']})")
        print(f"大小: {loc['width']} x {loc['height']}")
        print(f"中心: ({loc['center_x']}, {loc['center_y']})")

    if 'visualization' in result:
        cv2.imwrite('template_match_result.png', result['visualization'])
        print("结果已保存到 template_match_result.png")


if __name__ == "__main__":
    main()
