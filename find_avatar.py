#!/usr/bin/env python3
"""
头像查找工具 - 在聊天截图中定位头像

用法:
    python find_avatar.py <头像图片> <聊天截图> [输出图片]

示例:
    python find_avatar.py photo2.png msg.png avatar_found.png
"""
import sys
import cv2
import numpy as np


def find_avatar(avatar_path, chat_path, output_path="avatar_found.png"):
    avatar = cv2.imread(avatar_path)
    chat = cv2.imread(chat_path)

    if avatar is None:
        raise ValueError(f"无法读取头像: {avatar_path}")
    if chat is None:
        raise ValueError(f"无法读取聊天截图: {chat_path}")

    ah, aw = avatar.shape[:2]
    ch, cw = chat.shape[:2]

    best_score = 0
    best_loc = None
    best_size = (0, 0)

    # 多尺度模板匹配（彩色）
    # 聊天头像通常很小，从极小尺寸开始扫
    for scale in np.arange(0.03, 1.1, 0.01):
        new_w, new_h = int(aw * scale), int(ah * scale)
        if new_w < 8 or new_h < 8 or new_w > cw or new_h > ch:
            continue

        resized = cv2.resize(avatar, (new_w, new_h))

        # 用彩色匹配（保留紫色光圈等颜色特征）
        res = cv2.matchTemplate(chat, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_size = (new_w, new_h)

    if best_loc is None:
        print("❌ 未找到匹配")
        return False

    x, y = best_loc
    w, h = best_size

    result_img = chat.copy()
    color = (0, 255, 0) if best_score > 0.4 else (0, 165, 255)
    cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(result_img, (cx, cy), min(w, h) // 2, color, 2)
    label = f"score={best_score:.2f} size={w}x{h}"
    cv2.putText(result_img, label, (x, max(y - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imwrite(output_path, result_img)

    status = "✅ 找到" if best_score > 0.4 else "⚠️ 低置信度"
    print(f"{status} 头像位置: ({x}, {y})  大小: {w}x{h}  分数: {best_score:.2%}")
    print(f"✓ 结果已保存: {output_path}")
    return best_score > 0.4


def batch_find():
    import os
    import glob

    script_dir = os.path.dirname(os.path.abspath(__file__))
    thumb_path = os.path.join(script_dir, "thumb.png")
    example_dir = os.path.join(script_dir, "example")

    if not os.path.exists(thumb_path):
        print(f"❌ 找不到 thumb.png: {thumb_path}")
        sys.exit(1)

    patterns = [os.path.join(example_dir, "*.png"), os.path.join(example_dir, "*.jpg")]
    chat_files = []
    for p in patterns:
        chat_files.extend(glob.glob(p))

    # 过滤掉结果文件（_res 结尾）
    chat_files = [f for f in chat_files if not os.path.basename(f).endswith("_res.png")]
    chat_files.sort()

    if not chat_files:
        print(f"❌ example/ 目录下没有找到图片")
        sys.exit(1)

    import time

    print(f"头像: {thumb_path}")
    print(f"共找到 {len(chat_files)} 张图片待处理\n")

    total_start = time.time()
    for chat_path in chat_files:
        basename = os.path.splitext(os.path.basename(chat_path))[0]
        output_path = os.path.join(example_dir, f"{basename}_res.png")
        print(f"处理: {os.path.basename(chat_path)} -> {os.path.basename(output_path)}")
        t0 = time.time()
        find_avatar(thumb_path, chat_path, output_path)
        print(f"耗时: {time.time() - t0:.2f}s\n")

    print(f"全部完成，总耗时: {time.time() - total_start:.2f}s")


def main():
    batch_find()


if __name__ == "__main__":
    main()
