"""
rps_game.py
===========
بازی دو نفره‌ی سنگ کاغذ قیچی با تشخیص دقیق دست (MediaPipe) و مدلی که
خودت با train_gestures.py آموزش داده‌ای. طراحی به‌صورت پنل‌های شیشه‌ای
تیره با رنگ‌بندی نئونی، آیکون‌های برداری برای هر حرکت، نوار اطمینان
مدل، خط جداکننده‌ی درخشان بین دو بازیکن، و افکت فلش هنگام برد/تساوی.

پیش‌نیاز: قبل از اجرای این فایل حتماً یک‌بار train_gestures.py را اجرا
و مدل را بساز (فایل gesture_model.pkl باید کنار این اسکریپت باشد).

اجرا:
    python rps_game.py

کنترل‌ها:
    SPACE    -> شروع دور جدید (شمارش معکوس 3-2-1)
    R        -> ریست امتیازها
    Q / ESC  -> خروج
"""

import os
import time
import numpy as np
import cv2
import joblib

from hand_utils import HandTracker, draw_hand_skeleton
import ui_kit as ui

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_model.pkl")
CONFIDENCE_THRESHOLD = 0.55
BEATS = {"ROCK": "SCISSORS", "SCISSORS": "PAPER", "PAPER": "ROCK"}

STATE_IDLE, STATE_COUNTDOWN, STATE_RESULT, STATE_NO_MODEL = "IDLE", "COUNTDOWN", "RESULT", "NO_MODEL"


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    data = joblib.load(MODEL_PATH)
    return data["model"], data["labels"]


def classify(model, labels, features):
    """برمی‌گرداند (gesture یا None, confidence 0..1)."""
    proba = model.predict_proba([features])[0]
    idx = int(np.argmax(proba))
    conf = float(proba[idx])
    if conf < CONFIDENCE_THRESHOLD:
        return None, conf
    return labels[idx], conf


def decide_winner(g1, g2):
    if g1 is None or g2 is None:
        return "INVALID"
    if g1 == g2:
        return "TIE"
    return "P1" if BEATS[g1] == g2 else "P2"


def assign_hands_to_players(hands, frame_w):
    """هر دست را بر اساس موقعیت افقی به بازیکن ۱ (چپ) یا ۲ (راست) نسبت می‌دهد."""
    p1_hand, p2_hand = None, None
    for hand in hands:
        if hand["center_x"] < frame_w / 2:
            if p1_hand is None or hand["center_x"] < p1_hand["center_x"]:
                p1_hand = hand
        else:
            if p2_hand is None or hand["center_x"] > p2_hand["center_x"]:
                p2_hand = hand
    return p1_hand, p2_hand


def draw_player_panel(frame, x_center, color, name, gesture, confidence, is_left):
    """پنل بالای هر بازیکن: نام، آیکون حرکتِ لحظه‌ای، نوار اطمینان."""
    w_panel = 190
    x1 = int(x_center - w_panel / 2)
    x2 = int(x_center + w_panel / 2)
    ui.rounded_rect(frame, (x1, 96), (x2, 210), ui.PANEL, radius=18, alpha=0.72)
    ui.centered_text(frame, name, x_center, 122, 0.6, color, 2)

    icon_center = (int(x_center), 158)
    if gesture:
        ui.draw_gesture_icon(frame, gesture, icon_center, 22, color)
        label = gesture
    else:
        ui.centered_text(frame, "?", x_center, 168, 1.1, ui.GRAY, 2)
        label = "..."
    ui.centered_text(frame, label, x_center, 190, 0.5, ui.WHITE, 1)
    ui.progress_bar(frame, (x1 + 15, 198), (x2 - 15, 206), confidence, color)


def main():
    model, labels = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("خطا: وب‌کم پیدا نشد.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker(max_hands=2)

    score1, score2, round_no = 0, 0, 1
    state = STATE_IDLE if model is not None else STATE_NO_MODEL
    countdown_start = 0
    result_start = 0
    last_result_text = ""
    last_g1, last_g2 = None, None
    last_c1, last_c2 = 0.0, 0.0

    header_gradient = None
    cv2.namedWindow("Rock Paper Scissors", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if header_gradient is None:
            header_gradient = ui.gradient_bar(w, 96, ui.CYAN_P1, ui.MAGENTA_P2)

        # کمی تیره کردن کل فریم برای تضاد بهتر با رنگ‌های نئونی رابط کاربری
        dark_overlay = frame.copy()
        cv2.addWeighted(dark_overlay, 0.82, np.zeros_like(frame), 0.18, 0, frame)

        # خط جداکننده‌ی نقطه‌چین وسط صفحه
        ui.dashed_vline(frame, w // 2, 96, h - 50, ui.GRAY, dash=16, gap=12, thickness=2)

        if state == STATE_NO_MODEL:
            ui.rounded_rect(frame, (0, 0), (w, h), ui.BG_DARK, radius=0, alpha=0.55)
            ui.centered_text(frame, "MODEL NOT FOUND!", w // 2, h // 2 - 30, 1.0, ui.RED, 2)
            ui.centered_text(frame, "Run train_gestures.py first to build your model", w // 2, h // 2 + 15, 0.6, ui.WHITE, 1)
            cv2.imshow("Rock Paper Scissors", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            continue

        hands = tracker.process(frame)
        p1_hand, p2_hand = assign_hands_to_players(hands, w)

        gesture1 = conf1 = None
        gesture2 = conf2 = None
        if p1_hand is not None:
            draw_hand_skeleton(frame, p1_hand["landmarks_px"], ui.CYAN_P1)
            gesture1, conf1 = classify(model, labels, p1_hand["features"])
        if p2_hand is not None:
            draw_hand_skeleton(frame, p2_hand["landmarks_px"], ui.MAGENTA_P2)
            gesture2, conf2 = classify(model, labels, p2_hand["features"])
        conf1 = conf1 or 0.0
        conf2 = conf2 or 0.0

        # ---------------- هدر: گرادیان + امتیاز ----------------
        ui.blend_region(frame, header_gradient, (0, 0), (w, 96), alpha=0.22)
        ui.rounded_rect(frame, (0, 0), (w, 96), ui.BG_DARK, radius=0, alpha=0.55)
        ui.centered_text(frame, f"ROUND {round_no}", w // 2, 30, 0.7, ui.GOLD, 2)
        score_text = f"{score1}   -   {score2}"
        ui.centered_text(frame, score_text, w // 2, 68, 1.1, ui.WHITE, 2)

        draw_player_panel(frame, w * 0.22, ui.CYAN_P1, "PLAYER 1", gesture1, conf1, True)
        draw_player_panel(frame, w * 0.78, ui.MAGENTA_P2, "PLAYER 2", gesture2, conf2, False)

        # ---------------- ماشین‌حالت ----------------
        if state == STATE_IDLE:
            footer = "SPACE: New Round   |   R: Reset Score   |   Q: Quit"
            ui.rounded_rect(frame, (0, h - 44), (w, h), ui.BG_DARK, radius=0, alpha=0.6)
            ui.centered_text(frame, footer, w // 2, h - 15, 0.5, ui.WHITE, 1)
            ui.centered_text(frame, "Press SPACE to start", w // 2, h - 60, 0.8, ui.GOLD, 2)

        elif state == STATE_COUNTDOWN:
            elapsed = time.time() - countdown_start
            remaining = 3 - int(elapsed)
            cx, cy = w // 2, h // 2
            pulse = ui.pulse_value(time.time(), speed=6.0, lo=46, hi=60)
            cv2.circle(frame, (cx, cy), int(pulse), ui.GOLD, 4, cv2.LINE_AA)
            if remaining > 0:
                ui.centered_text(frame, str(remaining), cx, cy + 20, 2.2, ui.GOLD, 6)
            else:
                last_g1, last_g2 = gesture1, gesture2
                last_c1, last_c2 = conf1, conf2
                winner = decide_winner(last_g1, last_g2)
                if winner == "P1":
                    score1 += 1
                    last_result_text = "PLAYER 1 WINS!"
                elif winner == "P2":
                    score2 += 1
                    last_result_text = "PLAYER 2 WINS!"
                elif winner == "TIE":
                    last_result_text = "TIE!"
                else:
                    last_result_text = "DETECT FAILED - TRY AGAIN"
                round_no += 1
                state = STATE_RESULT
                result_start = time.time()

        elif state == STATE_RESULT:
            cx, cy = w // 2, h // 2
            panel_w, panel_h = 420, 220
            ui.rounded_rect(frame, (cx - panel_w // 2, cy - panel_h // 2),
                             (cx + panel_w // 2, cy + panel_h // 2), ui.PANEL, radius=24, alpha=0.85)

            g1 = last_g1 or "?"
            g2 = last_g2 or "?"
            if last_g1:
                ui.draw_gesture_icon(frame, last_g1, (cx - 90, cy - 30), 30, ui.CYAN_P1)
            ui.centered_text(frame, "VS", cx, cy - 15, 0.9, ui.WHITE, 2)
            if last_g2:
                ui.draw_gesture_icon(frame, last_g2, (cx + 90, cy - 30), 30, ui.MAGENTA_P2)

            ui.centered_text(frame, g1, cx - 90, cy + 40, 0.55, ui.CYAN_P1, 1)
            ui.centered_text(frame, g2, cx + 90, cy + 40, 0.55, ui.MAGENTA_P2, 1)

            color = ui.GOLD if "TIE" in last_result_text or "FAILED" in last_result_text else ui.GREEN
            ui.centered_text(frame, last_result_text, cx, cy + 85, 0.85, color, 2)

            flash_color = ui.GREEN if "WINS" in last_result_text else (
                ui.GOLD if "TIE" in last_result_text else ui.RED
            )
            cv2.rectangle(frame, (4, 4), (w - 4, h - 4), flash_color, 6, cv2.LINE_AA)

            ui.rounded_rect(frame, (0, h - 44), (w, h), ui.BG_DARK, radius=0, alpha=0.6)
            ui.centered_text(frame, "SPACE: Next Round   |   R: Reset   |   Q: Quit", w // 2, h - 15, 0.5, ui.WHITE, 1)

            if time.time() - result_start > 3.5:
                state = STATE_IDLE

        cv2.imshow("Rock Paper Scissors", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' ') and state in (STATE_IDLE, STATE_RESULT):
            state = STATE_COUNTDOWN
            countdown_start = time.time()
        elif key == ord('r'):
            score1, score2, round_no = 0, 0, 1
            state = STATE_IDLE

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
