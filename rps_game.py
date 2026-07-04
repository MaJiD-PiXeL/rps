"""
Rock - Paper - Scissors : Two-Player Webcam Edition
====================================================
یک بازی دو نفره‌ی «سنگ کاغذ قیچی» که با یک وب‌کم و OpenCV اجرا می‌شود.
هر بازیکن دست خودش را داخل کادر اختصاصی خودش (چپ / راست تصویر) می‌گیرد،
برنامه با تشخیص رنگ پوست + Convex Hull / Convexity Defects حرکت دست را
تشخیص می‌دهد (مشت = سنگ، دست باز = کاغذ، دو انگشت = قیچی) و برنده‌ی هر
دور را اعلام می‌کند.

اجرا:
    pip install opencv-python numpy
    python rps_game.py

کنترل‌ها:
    C        -> کالیبره کردن رنگ پوست (دست‌ها را داخل مربع‌های کوچک بگذارید و C را بزنید)
    SPACE    -> شروع دور جدید (شمارش معکوس 3-2-1)
    R        -> ریست امتیازها
    Q / ESC  -> خروج
"""

import cv2
import numpy as np
import time
import random

# ---------------------------------------------------------------------------
#  تنظیمات و پالت رنگی (ظاهر بازی)
# ---------------------------------------------------------------------------

WINDOW_NAME = "Rock Paper Scissors - 2 Player"

COLOR_BG_DARK   = (25, 20, 20)      # پس‌زمینه‌ی تیره‌ی نوار بالا/پایین
COLOR_P1        = (255, 190, 0)     # آبی-فیروزه‌ای (نارنجی-آبی در BGR) برای بازیکن ۱
COLOR_P2        = (170, 60, 255)    # صورتی-بنفش برای بازیکن ۲
COLOR_WHITE     = (245, 245, 245)
COLOR_GOLD      = (0, 210, 255)
COLOR_GREEN     = (90, 220, 90)
COLOR_RED       = (60, 60, 230)
FONT            = cv2.FONT_HERSHEY_DUPLEX

GESTURES = ["ROCK", "PAPER", "SCISSORS"]

# قوانین بازی: کلید می‌برد مقدار
BEATS = {"ROCK": "SCISSORS", "SCISSORS": "PAPER", "PAPER": "ROCK"}

# محدوده‌ی پیش‌فرض رنگ پوست در فضای HSV (در صورت نیاز با کالیبراسیون تغییر می‌کند)
DEFAULT_LOWER_SKIN = np.array([0, 30, 60], dtype=np.uint8)
DEFAULT_UPPER_SKIN = np.array([25, 150, 255], dtype=np.uint8)


# ---------------------------------------------------------------------------
#  توابع کمکی ترسیم (برای ظاهر شیک‌تر)
# ---------------------------------------------------------------------------

def draw_text_shadow(img, text, org, scale, color, thickness=2, font=FONT):
    """متن را با یک سایه‌ی مشکی زیرش رسم می‌کند تا خواناتر و شیک‌تر باشد."""
    x, y = org
    cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_translucent_rect(img, pt1, pt2, color, alpha=0.35):
    """یک مستطیل نیمه‌شفاف روی تصویر می‌کشد (برای پنل‌ها و نوارهای اطلاعات)."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_corner_brackets(img, pt1, pt2, color, length=28, thickness=3):
    """به‌جای یک کادر ساده، فقط گوشه‌های کادر را می‌کشد؛ ظاهری مدرن‌تر می‌سازد."""
    x1, y1 = pt1
    x2, y2 = pt2
    corners = [
        ((x1, y1), (1, 0), (0, 1)),
        ((x2, y1), (-1, 0), (0, 1)),
        ((x1, y2), (1, 0), (0, -1)),
        ((x2, y2), (-1, 0), (0, -1)),
    ]
    for (x, y), dx, dy in corners:
        cv2.line(img, (x, y), (x + dx[0] * length, y + dy[1] * length if dx[0] == 0 else y), color, thickness)
        cv2.line(img, (x, y), (x + dx[0] * length, y), color, thickness)
        cv2.line(img, (x, y), (x, y + dy[1] * length), color, thickness)


def draw_scoreboard(img, w, score1, score2, round_no):
    """نوار بالای صفحه شامل نام بازیکن‌ها، امتیاز و شماره‌ی دور."""
    draw_translucent_rect(img, (0, 0), (w, 70), COLOR_BG_DARK, alpha=0.55)
    draw_text_shadow(img, f"PLAYER 1", (25, 30), 0.7, COLOR_P1)
    draw_text_shadow(img, f"{score1}", (25, 62), 0.9, COLOR_WHITE)

    draw_text_shadow(img, f"PLAYER 2", (w - 175, 30), 0.7, COLOR_P2)
    draw_text_shadow(img, f"{score2}", (w - 60, 62), 0.9, COLOR_WHITE)

    round_text = f"ROUND {round_no}"
    text_size = cv2.getTextSize(round_text, FONT, 0.7, 2)[0]
    draw_text_shadow(img, round_text, (w // 2 - text_size[0] // 2, 30), 0.7, COLOR_GOLD)


def draw_footer(img, w, h, text):
    draw_translucent_rect(img, (0, h - 40), (w, h), COLOR_BG_DARK, alpha=0.55)
    text_size = cv2.getTextSize(text, FONT, 0.55, 1)[0]
    draw_text_shadow(img, text, (w // 2 - text_size[0] // 2, h - 14), 0.55, COLOR_WHITE, thickness=1)


# ---------------------------------------------------------------------------
#  تشخیص حرکت دست در یک ناحیه (ROI)
# ---------------------------------------------------------------------------

class HandGestureDetector:
    """
    دست را داخل یک ناحیه‌ی مستطیلی (ROI) با ماسک رنگ پوست پیدا می‌کند،
    Convex Hull و Convexity Defects را حساب می‌کند و بر اساس تعداد
    "شکاف بین انگشت‌ها" حرکت را به یکی از سنگ/کاغذ/قیچی نگاشت می‌کند.
    """

    def __init__(self):
        self.lower_skin = DEFAULT_LOWER_SKIN.copy()
        self.upper_skin = DEFAULT_UPPER_SKIN.copy()

    def calibrate(self, hsv_roi, sample_box):
        """میانگین رنگ پوست را از یک مربع کوچک نمونه‌برداری و بازه را تنظیم می‌کند."""
        x1, y1, x2, y2 = sample_box
        sample = hsv_roi[y1:y2, x1:x2]
        if sample.size == 0:
            return
        mean = sample.reshape(-1, 3).mean(axis=0)
        h, s, v = mean
        self.lower_skin = np.array([max(h - 15, 0), max(s - 60, 20), max(v - 70, 30)], dtype=np.uint8)
        self.upper_skin = np.array([min(h + 15, 179), 255, 255], dtype=np.uint8)

    def _mask(self, roi_bgr):
        blurred = cv2.GaussianBlur(roi_bgr, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask

    def detect(self, roi_bgr):
        """
        خروجی: (gesture_name یا None, mask برای دیباگ/نمایش, contour یا None)
        """
        mask = self._mask(roi_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask, None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        roi_area = roi_bgr.shape[0] * roi_bgr.shape[1]

        # اگر ناحیه خیلی کوچک باشد یعنی دستی داخل کادر نیست
        if area < roi_area * 0.04:
            return None, mask, None

        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 4:
            return "ROCK", mask, contour

        try:
            defects = cv2.convexityDefects(contour, hull_indices)
        except cv2.error:
            return "ROCK", mask, contour

        if defects is None:
            return "ROCK", mask, contour

        finger_gaps = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = np.linalg.norm(np.array(end) - np.array(start))
            b = np.linalg.norm(np.array(far) - np.array(start))
            c = np.linalg.norm(np.array(end) - np.array(far))
            if b * c == 0:
                continue
            angle = np.degrees(np.arccos(np.clip((b ** 2 + c ** 2 - a ** 2) / (2 * b * c), -1, 1)))

            # فقط شکاف‌های تیز بین انگشت‌ها (زاویه کوچک) و با عمق کافی را می‌شماریم
            if angle <= 90 and d > 9000:
                finger_gaps += 1

        # نگاشت تعداد شکاف بین انگشت‌ها به حرکت بازی
        if finger_gaps == 0:
            gesture = "ROCK"
        elif finger_gaps == 1:
            gesture = "SCISSORS"
        elif finger_gaps >= 3:
            gesture = "PAPER"
        else:
            # حالت مبهم (مثلاً ۲ شکاف) - نزدیک‌ترین حدس را برمی‌گردانیم
            gesture = "SCISSORS" if finger_gaps == 2 else "PAPER"

        return gesture, mask, contour


# ---------------------------------------------------------------------------
#  منطق تعیین برنده
# ---------------------------------------------------------------------------

def decide_winner(g1, g2):
    if g1 is None or g2 is None:
        return "INVALID"
    if g1 == g2:
        return "TIE"
    if BEATS[g1] == g2:
        return "P1"
    return "P2"


# ---------------------------------------------------------------------------
#  حلقه‌ی اصلی بازی
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("خطا: وب‌کم پیدا نشد. مطمئن شو دوربین لپ‌تاپ آزاد و در دسترس است.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector_p1 = HandGestureDetector()
    detector_p2 = HandGestureDetector()

    score1, score2, round_no = 0, 0, 1

    STATE_IDLE, STATE_COUNTDOWN, STATE_RESULT = "IDLE", "COUNTDOWN", "RESULT"
    state = STATE_IDLE
    countdown_start = 0
    result_start = 0
    last_result_text = ""
    last_g1, last_g2 = None, None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # --- تعریف ناحیه‌ی هر بازیکن (چپ برای پلیر۱، راست برای پلیر۲) ---
        box_w, box_h = int(w * 0.28), int(h * 0.55)
        top = int(h * 0.22)

        p1_box = (int(w * 0.04), top, int(w * 0.04) + box_w, top + box_h)
        p2_box = (w - int(w * 0.04) - box_w, top, w - int(w * 0.04), top + box_h)

        roi1 = frame[p1_box[1]:p1_box[3], p1_box[0]:p1_box[2]]
        roi2 = frame[p2_box[1]:p2_box[3], p2_box[0]:p2_box[2]]

        gesture1, mask1, contour1 = detector_p1.detect(roi1)
        gesture2, mask2, contour2 = detector_p2.detect(roi2)

        # کانتور دست را داخل کادر مربوطه بکش (برای فیدبک بصری زنده)
        if contour1 is not None:
            cv2.drawContours(roi1, [contour1], -1, COLOR_P1, 2)
        if contour2 is not None:
            cv2.drawContours(roi2, [contour2], -1, COLOR_P2, 2)

        # کادرهای شیک با گوشه‌های جدا
        draw_corner_brackets(frame, (p1_box[0], p1_box[1]), (p1_box[2], p1_box[3]), COLOR_P1)
        draw_corner_brackets(frame, (p2_box[0], p2_box[1]), (p2_box[2], p2_box[3]), COLOR_P2)

        # مربع‌های کوچک کالیبراسیون (وسط هر کادر)
        cal_size = 40
        p1_cal = (box_w // 2 - cal_size // 2, box_h // 2 - cal_size // 2,
                  box_w // 2 + cal_size // 2, box_h // 2 + cal_size // 2)
        p2_cal = p1_cal
        if state == STATE_IDLE:
            cv2.rectangle(roi1, (p1_cal[0], p1_cal[1]), (p1_cal[2], p1_cal[3]), COLOR_GOLD, 2)
            cv2.rectangle(roi2, (p2_cal[0], p2_cal[1]), (p2_cal[2], p2_cal[3]), COLOR_GOLD, 2)

        # برچسب زنده‌ی حرکت تشخیص داده شده زیر هر کادر
        label1 = gesture1 if gesture1 else "..."
        label2 = gesture2 if gesture2 else "..."
        draw_text_shadow(frame, label1, (p1_box[0], p1_box[3] + 30), 0.8, COLOR_P1)
        t2size = cv2.getTextSize(label2, FONT, 0.8, 2)[0]
        draw_text_shadow(frame, label2, (p2_box[2] - t2size[0], p2_box[3] + 30), 0.8, COLOR_P2)

        # --- ماشین‌حالت بازی ---
        if state == STATE_IDLE:
            draw_footer(frame, w, h, "C: Calibrate skin color   |   SPACE: Start round   |   R: Reset   |   Q: Quit")
            msg = "GET READY - Press SPACE to start"
            tsize = cv2.getTextSize(msg, FONT, 0.9, 2)[0]
            draw_text_shadow(frame, msg, (w // 2 - tsize[0] // 2, h - 60), 0.9, COLOR_WHITE)

        elif state == STATE_COUNTDOWN:
            elapsed = time.time() - countdown_start
            remaining = 3 - int(elapsed)
            if remaining > 0:
                text = str(remaining)
                tsize = cv2.getTextSize(text, FONT, 4, 8)[0]
                draw_text_shadow(frame, text, (w // 2 - tsize[0] // 2, h // 2), 4, COLOR_GOLD, thickness=8)
            else:
                # لحظه‌ی ضبط حرکت
                last_g1, last_g2 = gesture1, gesture2
                winner = decide_winner(last_g1, last_g2)
                if winner == "P1":
                    score1 += 1
                    last_result_text = "PLAYER 1 WINS!"
                elif winner == "P2":
                    score2 += 1
                    last_result_text = "PLAYER 2 WINS!"
                elif winner == "TIE":
                    last_result_text = "IT'S A TIE!"
                else:
                    last_result_text = "COULD NOT DETECT - TRY AGAIN"
                round_no += 1
                state = STATE_RESULT
                result_start = time.time()

        elif state == STATE_RESULT:
            g1_text = last_g1 if last_g1 else "?"
            g2_text = last_g2 if last_g2 else "?"
            reveal = f"{g1_text}   VS   {g2_text}"
            tsize = cv2.getTextSize(reveal, FONT, 1.0, 2)[0]
            draw_translucent_rect(frame, (w // 2 - tsize[0] // 2 - 20, h // 2 - 80),
                                   (w // 2 + tsize[0] // 2 + 20, h // 2 + 40), COLOR_BG_DARK, alpha=0.6)
            draw_text_shadow(frame, reveal, (w // 2 - tsize[0] // 2, h // 2 - 40), 1.0, COLOR_WHITE)

            color = COLOR_GREEN if "1" in last_result_text or "2" in last_result_text else COLOR_RED
            if "TIE" in last_result_text:
                color = COLOR_GOLD
            tsize2 = cv2.getTextSize(last_result_text, FONT, 0.9, 2)[0]
            draw_text_shadow(frame, last_result_text, (w // 2 - tsize2[0] // 2, h // 2), 0.9, color)

            draw_footer(frame, w, h, "SPACE: Next round   |   R: Reset   |   Q: Quit")

            if time.time() - result_start > 3.5:
                state = STATE_IDLE

        draw_scoreboard(frame, w, score1, score2, round_no)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q یا ESC
            break
        elif key == ord(' ') and state in (STATE_IDLE, STATE_RESULT):
            state = STATE_COUNTDOWN
            countdown_start = time.time()
        elif key == ord('r'):
            score1, score2, round_no = 0, 0, 1
            state = STATE_IDLE
        elif key == ord('c') and state == STATE_IDLE:
            hsv1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
            detector_p1.calibrate(hsv1, p1_cal)
            detector_p2.calibrate(hsv2, p2_cal)
            print("کالیبراسیون رنگ پوست انجام شد.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
