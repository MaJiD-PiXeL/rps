"""
ui_kit.py
=========
مجموعه‌ی توابع طراحی برای ظاهر یکپارچه و مدرن بین اسکریپت آموزش و بازی.
پالت تیره با رنگ‌های نئونی، پنل‌های شیشه‌ای نیمه‌شفاف، گوشه‌های گرد،
آیکون‌های برداری برای سنگ/کاغذ/قیچی (به‌جای فقط متن) و افکت‌های ساده‌ی
انیمیشنی (پالس، فلاش، گرادیان).
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
#  پالت رنگی (BGR چون OpenCV به این ترتیب کار می‌کند)
# ---------------------------------------------------------------------------

BG_DARK      = (28, 18, 18)      # #12121c پس‌زمینه‌ی تیره
PANEL        = (47, 27, 27)      # #1b1b2f پنل شیشه‌ای
CYAN_P1      = (255, 229, 0)     # #00e5ff  رنگ اختصاصی بازیکن ۱
MAGENTA_P2   = (166, 46, 255)    # #ff2ea6  رنگ اختصاصی بازیکن ۲
GOLD         = (102, 209, 255)   # #ffd166
GREEN        = (160, 214, 6)     # #06d6a0
RED          = (111, 71, 239)    # #ef476f
WHITE        = (247, 245, 245)   # #f5f5f7
GRAY         = (134, 112, 108)   # #6c7086

FONT = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------------------
#  اشکال پایه: پنل شیشه‌ای با گوشه‌ی گرد + متن با درخشش
# ---------------------------------------------------------------------------

def rounded_rect(img, pt1, pt2, color, radius=16, thickness=-1, alpha=1.0):
    """مستطیل با گوشه‌ی گرد؛ اگر alpha<1 باشد به‌صورت نیمه‌شفاف روی تصویر می‌نشیند."""
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    target = img.copy() if alpha < 1.0 else img

    if thickness < 0:
        cv2.rectangle(target, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(target, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [
            (x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius), (x2 - radius, y2 - radius),
        ]:
            cv2.circle(target, (cx, cy), radius, color, -1, cv2.LINE_AA)
    else:
        cv2.ellipse(target, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(target, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(target, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(target, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.line(target, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(target, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(target, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(target, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)

    if alpha < 1.0:
        cv2.addWeighted(target, alpha, img, 1 - alpha, 0, img)


def glow_text(img, text, org, scale, color, thickness=2, font=FONT):
    """متن با یک سایه‌ی مشکی که خوانایی روی تصویر دوربین را بالا می‌برد."""
    x, y = org
    cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def centered_text(img, text, cx, y, scale, color, thickness=2, font=FONT):
    size = cv2.getTextSize(text, font, scale, thickness)[0]
    glow_text(img, text, (int(cx - size[0] / 2), y), scale, color, thickness, font)
    return size


def gradient_bar(w, h, color_left, color_right):
    """یک نوار گرادیانی افقی می‌سازد (یک‌بار محاسبه و در حلقه استفاده شود)."""
    bar = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        bar[:, :, i] = np.linspace(color_left[i], color_right[i], w, dtype=np.uint8)
    return bar


def blend_region(img, overlay, pt1, pt2, alpha=0.85):
    x1, y1 = pt1
    x2, y2 = pt2
    region = img[y1:y2, x1:x2]
    if region.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (region.shape[1], region.shape[0]))
    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, region)
    img[y1:y2, x1:x2] = region


def dashed_vline(img, x, y1, y2, color, dash=14, gap=10, thickness=3):
    y = y1
    while y < y2:
        y_end = min(y + dash, y2)
        cv2.line(img, (x, y), (x, y_end), color, thickness, cv2.LINE_AA)
        y += dash + gap


def progress_bar(img, pt1, pt2, ratio, color, bg_color=PANEL):
    """نوار پیشرفت افقی (برای نمایش درصد اطمینان مدل)."""
    ratio = max(0.0, min(1.0, ratio))
    x1, y1 = pt1
    x2, y2 = pt2
    rounded_rect(img, (x1, y1), (x2, y2), bg_color, radius=(y2 - y1) // 2)
    fill_x2 = x1 + int((x2 - x1) * ratio)
    if fill_x2 > x1 + 2:
        rounded_rect(img, (x1, y1), (fill_x2, y2), color, radius=(y2 - y1) // 2)


def pulse_value(t, speed=4.0, lo=0.6, hi=1.0):
    """مقداری نوسانی بین lo و hi برای افکت پالس (مثلاً شعاع دایره‌ی شمارش‌معکوس)."""
    s = (np.sin(t * speed) + 1) / 2
    return lo + (hi - lo) * s


# ---------------------------------------------------------------------------
#  آیکون‌های برداری سنگ / کاغذ / قیچی
# ---------------------------------------------------------------------------

def icon_rock(img, center, r, color, thickness=4):
    cx, cy = center
    cv2.circle(img, (cx, cy), r, color, thickness, cv2.LINE_AA)
    # چند خط کوچک داخل، شبیه بند انگشت‌های مشت‌شده
    for dx in (-r // 2, 0, r // 2):
        cv2.line(img, (cx + dx, cy - r // 3), (cx + dx, cy + r // 4), color, max(2, thickness - 1), cv2.LINE_AA)


def icon_paper(img, center, size, color, thickness=4):
    cx, cy = center
    x1, y1 = cx - size, cy - size
    x2, y2 = cx + size, cy + size
    rounded_rect(img, (x1, y1), (x2, y2), color, radius=size // 4, thickness=thickness)
    # خطوط انگشت‌ها
    for i in range(1, 4):
        fx = x1 + i * (x2 - x1) // 4
        cv2.line(img, (fx, y1), (fx, y1 - size // 3), color, max(2, thickness - 2), cv2.LINE_AA)


def icon_scissors(img, center, size, color, thickness=4):
    cx, cy = center
    cv2.line(img, (cx, cy), (cx - size, cy - size), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy), (cx - size, cy + size), color, thickness, cv2.LINE_AA)
    cv2.circle(img, (cx - size, cy - size), size // 4, color, thickness - 1, cv2.LINE_AA)
    cv2.circle(img, (cx - size, cy + size), size // 4, color, thickness - 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)


def icon_trophy(img, center, size, color, thickness=3):
    """آیکون برداری جام قهرمانی برای اسکوربورد و صفحه‌ی پایان بازی."""
    cx, cy = center
    cup_w = size
    cup_top = cy - size
    cup_bottom = cy
    # بدنه‌ی جام (ذوزنقه)
    pts = np.array([
        [cx - cup_w, cup_top], [cx + cup_w, cup_top],
        [cx + cup_w // 2, cup_bottom], [cx - cup_w // 2, cup_bottom],
    ], np.int32)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
    # دسته‌های جام
    cv2.ellipse(img, (cx - cup_w - size // 4, cup_top + size // 3), (size // 4, size // 3), 0, -90, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (cx + cup_w + size // 4, cup_top + size // 3), (size // 4, size // 3), 0, 90, 270, color, thickness, cv2.LINE_AA)
    # پایه
    stem_w = size // 6
    cv2.line(img, (cx, cup_bottom), (cx, cup_bottom + size // 2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx - size // 2, cup_bottom + size // 2), (cx + size // 2, cup_bottom + size // 2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx - size // 2, cup_bottom + size // 2 + 6), (cx + size // 2, cup_bottom + size // 2 + 6), color, thickness, cv2.LINE_AA)


def _measure_pill(key_label, action_label):
    font_scale = 0.5
    key_size = cv2.getTextSize(key_label, FONT, font_scale, 1)[0]
    pad_x = 10
    key_w = key_size[0] + pad_x * 2
    action_size = cv2.getTextSize(action_label, FONT, font_scale, 1)[0]
    return key_w + 8 + action_size[0] + 24


def hotkey_bar(img, y, items, center_x):
    """items: لیستی از (key_label, action_label, color). دکمه‌ها را وسط‌چین کنار هم می‌چیند."""
    widths = [_measure_pill(k, a) for k, a, _ in items]
    total = sum(widths) - 24
    x = int(center_x - total / 2)
    for (k, a, c), wdt in zip(items, widths):
        key_pill(img, x, y, k, a, color=c)
        x += wdt


def key_pill(img, x, y, key_label, action_label, color=WHITE, key_color=None):
    """یک 'دکمه‌ی کلید' شبیه کیبورد + توضیح کوتاه کنارش؛ برای نوار راهنمای کلیدها.
    خروجی: عرض کل ناحیه‌ی رسم‌شده (برای چیدن پیل بعدی کنار آن)."""
    key_color = key_color or color
    font_scale = 0.5
    key_size = cv2.getTextSize(key_label, FONT, font_scale, 1)[0]
    pad_x, pad_y = 10, 6
    key_w = key_size[0] + pad_x * 2
    key_h = key_size[1] + pad_y * 2
    rounded_rect(img, (x, y), (x + key_w, y + key_h), PANEL, radius=8, alpha=0.9)
    rounded_rect(img, (x, y), (x + key_w, y + key_h), key_color, radius=8, thickness=1)
    glow_text(img, key_label, (x + pad_x, y + key_h - pad_y), font_scale, key_color, 1)

    action_size = cv2.getTextSize(action_label, FONT, font_scale, 1)[0]
    ax = x + key_w + 8
    glow_text(img, action_label, (ax, y + key_h - pad_y), font_scale, color, 1)
    return key_w + 8 + action_size[0] + 24  # فاصله تا آیتم بعدی


ICONS = {"ROCK": icon_rock, "PAPER": icon_paper, "SCISSORS": icon_scissors}
LABELS_FA = {"ROCK": "SANG", "PAPER": "KAGHAZ", "SCISSORS": "GHEYCHI"}


def draw_gesture_icon(img, gesture, center, size, color):
    fn = ICONS.get(gesture)
    if fn:
        if gesture == "ROCK":
            fn(img, center, size, color)
        else:
            fn(img, center, size, color)
    else:
        centered_text(img, "?", center[0], center[1] + size // 2, 1.4, color, 3)
