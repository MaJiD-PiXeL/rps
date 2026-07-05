"""
rps_game.py
===========
بازی دو نفره‌ی سنگ کاغذ قیچی با تشخیص دقیق دست (MediaPipe) و مدلی که
خودت داخل همین برنامه آموزش می‌دهی — دیگر نیازی به اجرای فایل جداگانه
نیست. اگر مدلی وجود نداشته باشد، برنامه با یک صفحه‌ی راهنمای کامل
قدم‌به‌قدم شروع می‌شود و می‌توانی همان‌جا با کلید T وارد حالت تمرین شوی.

اجرا:
    pip install opencv-python mediapipe scikit-learn joblib numpy
    python rps_game.py

کنترل‌های کلی (بسته به حالت فعلی، نوار پایین صفحه هم راهنما را نشان می‌دهد):
    T          -> ورود به حالت تمرین دوباره (هر وقت بخواهی مدل را بهتر کنی)
    SPACE      -> شروع دور جدید / دور بعد / مسابقه‌ی جدید
    R          -> ریست کامل امتیاز و شروع مسابقه‌ی جدید
    Q / ESC    -> خروج (در حالت تمرین: ESC فقط از تمرین خارج می‌شود)

کنترل‌های حالت تمرین:
    1 / 2 / 3   -> ثبت نمونه‌ی ROCK / PAPER / SCISSORS از دست فعلی
    BACKSPACE   -> حذف آخرین نمونه‌ی ثبت‌شده
    G           -> آموزش مدل با نمونه‌های جمع‌آوری‌شده و ذخیره روی دیسک
    ESC         -> لغو و بازگشت
"""

import os
import time
import numpy as np
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from hand_utils import HandTracker, draw_hand_skeleton, extract_features, augment_landmarks
import ui_kit as ui

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_model.pkl")
CONFIDENCE_THRESHOLD = 0.55
BEATS = {"ROCK": "SCISSORS", "SCISSORS": "PAPER", "PAPER": "ROCK"}
LABELS = ["ROCK", "PAPER", "SCISSORS"]
KEY_TO_LABEL = {ord('1'): "ROCK", ord('2'): "PAPER", ord('3'): "SCISSORS"}
MIN_SAMPLES_PER_CLASS = 15
RECOMMENDED_SAMPLES = 40
TARGET_SCORE = 5  # اولین کسی که به این امتیاز برسد، برنده‌ی مسابقه است

STATE_NO_MODEL = "NO_MODEL"
STATE_IDLE = "IDLE"
STATE_COUNTDOWN = "COUNTDOWN"
STATE_RESULT = "RESULT"
STATE_TRAIN = "TRAIN"
STATE_GAME_OVER = "GAME_OVER"

HEADER_H = 132
FOOTER_H = 58


# ---------------------------------------------------------------------------
#  توابع کمکی مدل
# ---------------------------------------------------------------------------

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    data = joblib.load(MODEL_PATH)
    return data["model"], data["labels"]


def classify(model, labels, features):
    proba = model.predict_proba([features])[0]
    idx = int(np.argmax(proba))
    conf = float(proba[idx])
    if conf < CONFIDENCE_THRESHOLD:
        return None, conf
    return labels[idx], conf


def get_probabilities(model, labels, features):
    """برای حالت دیباگ: احتمال هر سه کلاس را برمی‌گرداند، مثلاً {'ROCK':0.1,...}."""
    proba = model.predict_proba([features])[0]
    return {label: float(p) for label, p in zip(labels, proba)}


def decide_winner(g1, g2):
    if g1 is None or g2 is None:
        return "INVALID"
    if g1 == g2:
        return "TIE"
    return "P1" if BEATS[g1] == g2 else "P2"


def assign_hands_to_players(hands, frame_w):
    p1_hand, p2_hand = None, None
    for hand in hands:
        if hand["center_x"] < frame_w / 2:
            if p1_hand is None or hand["center_x"] < p1_hand["center_x"]:
                p1_hand = hand
        else:
            if p2_hand is None or hand["center_x"] > p2_hand["center_x"]:
                p2_hand = hand
    return p1_hand, p2_hand


def train_and_save_model(dataset):
    """
    dataset[label] لیستی از نمونه‌های خام لندمارک (21,3) هر کدام است.
    هر نمونه‌ی خام با augment_landmarks به چند نسخه (اصلی + آینه‌ای + نویزی)
    تبدیل و سپس با extract_features به بردار ویژگی نهایی تبدیل می‌شود؛
    این کار باعث می‌شود مدل خیلی بهتر از تنوع واقعیِ حین بازی (زاویه‌ی کمی
    متفاوت، دست چپ/راست بازیکن دوم، نویز دوربین) سربلند بیرون بیاید.
    """
    X, y = [], []
    for label, raw_samples in dataset.items():
        for raw_pts in raw_samples:
            for variant in augment_landmarks(raw_pts):
                X.append(extract_features(variant))
                y.append(label)
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=300, max_depth=16, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test)) if len(X_test) > 0 else float("nan")

    final_clf = RandomForestClassifier(n_estimators=300, max_depth=16, random_state=42)
    final_clf.fit(X, y)
    joblib.dump({"model": final_clf, "labels": LABELS}, MODEL_PATH)
    return final_clf, LABELS, acc


# ---------------------------------------------------------------------------
#  رسم اسکوربورد و نوار راهنما (همیشه در دسترس)
# ---------------------------------------------------------------------------

def draw_scoreboard(frame, w, score1, score2, round_no, title):
    """اسکوربورد اصلی بالای صفحه: نام بازیکن‌ها، امتیاز بزرگ، جام برای نفر جلوتر، دور فعلی."""
    ui.rounded_rect(frame, (14, 12), (w - 14, HEADER_H), ui.PANEL, radius=20, alpha=0.82)
    ui.rounded_rect(frame, (14, 12), (w - 14, HEADER_H), ui.GOLD, radius=20, thickness=2)

    cx = w // 2
    ui.centered_text(frame, title, cx, 42, 0.75, ui.GOLD, 2)

    # امتیاز بزرگ وسط
    score_text = f"{score1}   :   {score2}"
    ui.centered_text(frame, score_text, cx, 90, 1.5, ui.WHITE, 3)
    ui.centered_text(frame, f"FIRST TO {TARGET_SCORE} WINS THE MATCH", cx, 115, 0.42, ui.GRAY, 1)

    # نام بازیکن‌ها + جام برای جلوتر بودن
    ui.centered_text(frame, "PLAYER 1", int(w * 0.16), 45, 0.62, ui.CYAN_P1, 2)
    ui.centered_text(frame, "PLAYER 2", int(w * 0.84), 45, 0.62, ui.MAGENTA_P2, 2)
    if score1 > score2:
        ui.icon_trophy(frame, (int(w * 0.16) - 70, 40), 12, ui.GOLD)
    elif score2 > score1:
        ui.icon_trophy(frame, (int(w * 0.84) + 70, 40), 12, ui.GOLD)


def draw_hotkeys(frame, w, h, items):
    """نوار راهنمای کلیدها که همیشه پایین صفحه و متناسب با حالت فعلی نشان داده می‌شود."""
    ui.rounded_rect(frame, (14, h - FOOTER_H), (w - 14, h - 10), ui.BG_DARK, radius=16, alpha=0.75)
    ui.hotkey_bar(frame, h - FOOTER_H + 16, items, w // 2)


def draw_player_panel(frame, x_center, color, name, gesture, confidence, debug_probs=None):
    w_panel = 190
    y1, y2 = HEADER_H + 14, HEADER_H + 128
    x1 = int(x_center - w_panel / 2)
    x2 = int(x_center + w_panel / 2)
    ui.rounded_rect(frame, (x1, y1), (x2, y2), ui.PANEL, radius=18, alpha=0.72)

    icon_center = (int(x_center), y1 + 45)
    if gesture:
        ui.draw_gesture_icon(frame, gesture, icon_center, 22, color)
        label = gesture
    else:
        ui.centered_text(frame, "?", x_center, y1 + 55, 1.1, ui.GRAY, 2)
        label = "..."
    ui.centered_text(frame, label, x_center, y1 + 78, 0.5, ui.WHITE, 1)
    ui.progress_bar(frame, (x1 + 15, y1 + 88), (x2 - 15, y1 + 98), confidence, color)
    ui.centered_text(frame, name, x_center, y2 - 8, 0.42, color, 1)

    if debug_probs:
        txt = "  ".join(f"{k[0]}:{int(v * 100)}" for k, v in debug_probs.items())
        ui.centered_text(frame, txt, x_center, y2 + 20, 0.42, ui.GOLD, 1)


# ---------------------------------------------------------------------------
#  صفحه‌ی راهنمای کامل وقتی مدلی وجود ندارد (Onboarding)
# ---------------------------------------------------------------------------

def draw_onboarding(frame, w, h):
    ui.rounded_rect(frame, (0, 0), (w, h), ui.BG_DARK, radius=0, alpha=0.55)

    panel_w, panel_h = min(760, w - 80), 420
    x1, y1 = w // 2 - panel_w // 2, h // 2 - panel_h // 2
    x2, y2 = x1 + panel_w, y1 + panel_h
    ui.rounded_rect(frame, (x1, y1), (x2, y2), ui.PANEL, radius=24, alpha=0.9)
    ui.rounded_rect(frame, (x1, y1), (x2, y2), ui.GOLD, radius=24, thickness=2)

    ui.centered_text(frame, "WELCOME! LET'S SET UP YOUR HAND GESTURES", w // 2, y1 + 45, 0.85, ui.GOLD, 2)
    ui.centered_text(frame, "No trained model yet — follow these steps:", w // 2, y1 + 75, 0.5, ui.WHITE, 1)

    steps = [
        ("1", "Press  [T]  to enter Training Mode right now."),
        ("2", "Show ROCK / PAPER / SCISSORS to the camera one at a time."),
        ("3", "Press  1 / 2 / 3  each time to save a sample of that gesture."),
        ("4", f"Collect at least {MIN_SAMPLES_PER_CLASS} samples per gesture ({RECOMMENDED_SAMPLES}+ recommended)."),
        ("5", "Press  [G]  to train your personal model and save it."),
        ("6", "Press  [SPACE]  to start playing with a friend!"),
    ]
    ly = y1 + 115
    for num, text in steps:
        ui.rounded_rect(frame, (x1 + 40, ly - 18), (x1 + 68, ly + 6), ui.GOLD, radius=8)
        ui.centered_text(frame, num, x1 + 54, ly, 0.5, ui.BG_DARK, 2)
        ui.glow_text(frame, text, (x1 + 85, ly + 3), 0.52, ui.WHITE, 1)
        ly += 44

    ui.centered_text(frame, "Tip: good, even lighting on your hand = much better accuracy.",
                      w // 2, y2 - 20, 0.42, ui.GRAY, 1)


# ---------------------------------------------------------------------------
#  حالت تمرین (Training Mode) درون‌برنامه‌ای
# ---------------------------------------------------------------------------

def draw_training_ui(frame, w, h, dataset, current_gesture_ok, flash_until, last_label, min_ok):
    ui.rounded_rect(frame, (0, 0), (w, HEADER_H), ui.BG_DARK, radius=0, alpha=0.7)
    ui.centered_text(frame, "TRAINING MODE", w // 2, 42, 0.85, ui.GOLD, 2)
    ui.centered_text(frame, "Show a gesture, then press 1 / 2 / 3 to save a sample of it",
                      w // 2, 75, 0.5, ui.WHITE, 1)

    hint_color = ui.GREEN if current_gesture_ok else ui.RED
    hint_text = "Hand detected - ready to capture" if current_gesture_ok else "No hand detected..."
    ui.centered_text(frame, hint_text, w // 2, 105, 0.5, hint_color, 1)

    panel_y = h - FOOTER_H - 150
    ui.rounded_rect(frame, (20, panel_y), (w - 20, h - FOOTER_H - 16), ui.PANEL, radius=18, alpha=0.78)
    colors = {"ROCK": ui.CYAN_P1, "PAPER": ui.GOLD, "SCISSORS": ui.MAGENTA_P2}
    keys_hint = {"ROCK": "[1]", "PAPER": "[2]", "SCISSORS": "[3]"}
    col_w = (w - 40) // 3
    for i, label in enumerate(LABELS):
        cx = 20 + col_w * i + col_w // 2
        count = len(dataset[label])
        ratio = min(count / RECOMMENDED_SAMPLES, 1.0)
        ui.draw_gesture_icon(frame, label, (cx, panel_y + 40), 24, colors[label])
        ui.centered_text(frame, f"{keys_hint[label]} {label}", cx, panel_y + 82, 0.6, colors[label], 1)
        ui.progress_bar(frame, (cx - 75, panel_y + 92), (cx + 75, panel_y + 105), ratio, colors[label])
        ui.centered_text(frame, f"{count} samples", cx, panel_y + 128, 0.48, ui.WHITE, 1)

    if time.time() < flash_until:
        cv2.rectangle(frame, (4, 4), (w - 4, h - 4), ui.GREEN, 8, cv2.LINE_AA)
        ui.centered_text(frame, f"{last_label} sample saved!", w // 2, h // 2, 0.9, ui.GREEN, 2)

    if not min_ok:
        msg = f"Need at least {MIN_SAMPLES_PER_CLASS} samples of EACH gesture before training"
        color = ui.GRAY
    else:
        msg = "Ready! Press [G] to train and save your model"
        color = ui.GREEN
    ui.centered_text(frame, msg, w // 2, panel_y - 14, 0.5, color, 1)


# ---------------------------------------------------------------------------
#  حلقه‌ی اصلی
# ---------------------------------------------------------------------------

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
    state_before_train = state
    show_debug = False

    countdown_start = 0
    result_start = 0
    last_result_text = ""
    last_g1, last_g2 = None, None

    # داده‌های حالت تمرین
    train_dataset = {label: [] for label in LABELS}
    last_captured_label = None
    flash_until = 0

    cv2.namedWindow("Rock Paper Scissors", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        dark_overlay = frame.copy()
        cv2.addWeighted(dark_overlay, 0.82, np.zeros_like(frame), 0.18, 0, frame)

        hands = tracker.process(frame)

        # ---------------- حالت تمرین ----------------
        if state == STATE_TRAIN:
            current_hand = hands[0] if hands else None
            if current_hand:
                draw_hand_skeleton(frame, current_hand["landmarks_px"], ui.GOLD, ui.WHITE)

            min_ok = all(len(v) >= MIN_SAMPLES_PER_CLASS for v in train_dataset.values())
            draw_training_ui(frame, w, h, train_dataset, current_hand is not None,
                              flash_until, last_captured_label, min_ok)

            hotkeys = [("1/2/3", "Save Sample", ui.WHITE), ("BKSP", "Undo", ui.WHITE)]
            if min_ok:
                hotkeys.append(("G", "Train & Save", ui.GREEN))
            hotkeys.append(("ESC", "Cancel", ui.RED))
            draw_hotkeys(frame, w, h, hotkeys)

            cv2.imshow("Rock Paper Scissors", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC - لغو تمرین
                state = state_before_train
            elif key in KEY_TO_LABEL and current_hand is not None:
                label = KEY_TO_LABEL[key]
                train_dataset[label].append(current_hand["landmarks_norm"])
                last_captured_label = label
                flash_until = time.time() + 0.3
            elif key == 8:  # Backspace
                if last_captured_label and train_dataset[last_captured_label]:
                    train_dataset[last_captured_label].pop()
            elif key == ord('g') and min_ok:
                model, labels, acc = train_and_save_model(train_dataset)
                print(f"مدل آموزش دید و ذخیره شد. دقت روی داده‌ی تست: {acc * 100:.1f}%")
                state = STATE_IDLE
            continue  # بقیه‌ی حلقه (بازی) در حالت تمرین اجرا نشود

        # ---------------- حالت بدون مدل (راهنما) ----------------
        if state == STATE_NO_MODEL:
            draw_onboarding(frame, w, h)
            draw_hotkeys(frame, w, h, [("T", "Start Training", ui.GOLD), ("Q", "Quit", ui.RED)])
            cv2.imshow("Rock Paper Scissors", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('t'):
                train_dataset = {label: [] for label in LABELS}
                state_before_train = STATE_NO_MODEL
                state = STATE_TRAIN
            continue

        # ---------------- از این‌جا به بعد: بازی واقعی (مدل موجود است) ----------------
        p1_hand, p2_hand = assign_hands_to_players(hands, w)
        gesture1 = conf1 = None
        gesture2 = conf2 = None
        probs1 = probs2 = None
        if p1_hand is not None:
            draw_hand_skeleton(frame, p1_hand["landmarks_px"], ui.CYAN_P1)
            gesture1, conf1 = classify(model, labels, p1_hand["features"])
            if show_debug:
                probs1 = get_probabilities(model, labels, p1_hand["features"])
        if p2_hand is not None:
            draw_hand_skeleton(frame, p2_hand["landmarks_px"], ui.MAGENTA_P2)
            gesture2, conf2 = classify(model, labels, p2_hand["features"])
            if show_debug:
                probs2 = get_probabilities(model, labels, p2_hand["features"])
        conf1 = conf1 or 0.0
        conf2 = conf2 or 0.0

        ui.dashed_vline(frame, w // 2, HEADER_H + 10, h - FOOTER_H - 10, ui.GRAY, dash=16, gap=12, thickness=2)

        title = f"ROUND {round_no}" if state != STATE_GAME_OVER else "MATCH OVER"
        draw_scoreboard(frame, w, score1, score2, round_no, title)
        draw_player_panel(frame, w * 0.22, ui.CYAN_P1, "PLAYER 1", gesture1, conf1, probs1)
        draw_player_panel(frame, w * 0.78, ui.MAGENTA_P2, "PLAYER 2", gesture2, conf2, probs2)

        if state == STATE_IDLE:
            draw_hotkeys(frame, w, h, [
                ("SPACE", "Start Round", ui.GOLD), ("T", "Retrain", ui.CYAN_P1),
                ("D", "Debug" if not show_debug else "Debug: ON", ui.GOLD if show_debug else ui.WHITE),
                ("R", "Reset", ui.WHITE), ("Q", "Quit", ui.RED),
            ])
            ui.centered_text(frame, "Press SPACE to start!", w // 2, h - FOOTER_H - 20, 0.7, ui.GOLD, 2)

        elif state == STATE_COUNTDOWN:
            elapsed = time.time() - countdown_start
            remaining = 3 - int(elapsed)
            cx, cy = w // 2, (HEADER_H + h - FOOTER_H) // 2
            pulse = ui.pulse_value(time.time(), speed=6.0, lo=46, hi=60)
            cv2.circle(frame, (cx, cy), int(pulse), ui.GOLD, 4, cv2.LINE_AA)
            draw_hotkeys(frame, w, h, [("Q", "Quit", ui.RED)])
            if remaining > 0:
                ui.centered_text(frame, str(remaining), cx, cy + 20, 2.2, ui.GOLD, 6)
            else:
                last_g1, last_g2 = gesture1, gesture2
                winner = decide_winner(last_g1, last_g2)
                if winner == "P1":
                    score1 += 1
                    last_result_text = "PLAYER 1 WINS THE ROUND!"
                elif winner == "P2":
                    score2 += 1
                    last_result_text = "PLAYER 2 WINS THE ROUND!"
                elif winner == "TIE":
                    last_result_text = "TIE!"
                else:
                    last_result_text = "DETECT FAILED - TRY AGAIN"
                round_no += 1
                result_start = time.time()
                if score1 >= TARGET_SCORE or score2 >= TARGET_SCORE:
                    state = STATE_GAME_OVER
                else:
                    state = STATE_RESULT

        elif state == STATE_RESULT:
            cx, cy = w // 2, (HEADER_H + h - FOOTER_H) // 2
            panel_w, panel_h = 420, 210
            ui.rounded_rect(frame, (cx - panel_w // 2, cy - panel_h // 2),
                             (cx + panel_w // 2, cy + panel_h // 2), ui.PANEL, radius=24, alpha=0.88)
            g1 = last_g1 or "?"
            g2 = last_g2 or "?"
            if last_g1:
                ui.draw_gesture_icon(frame, last_g1, (cx - 90, cy - 25), 28, ui.CYAN_P1)
            ui.centered_text(frame, "VS", cx, cy - 10, 0.85, ui.WHITE, 2)
            if last_g2:
                ui.draw_gesture_icon(frame, last_g2, (cx + 90, cy - 25), 28, ui.MAGENTA_P2)
            ui.centered_text(frame, g1, cx - 90, cy + 35, 0.5, ui.CYAN_P1, 1)
            ui.centered_text(frame, g2, cx + 90, cy + 35, 0.5, ui.MAGENTA_P2, 1)

            color = ui.GOLD if "TIE" in last_result_text or "FAILED" in last_result_text else ui.GREEN
            ui.centered_text(frame, last_result_text, cx, cy + 78, 0.8, color, 2)

            flash_color = ui.GREEN if "WINS" in last_result_text else (
                ui.GOLD if "TIE" in last_result_text else ui.RED)
            cv2.rectangle(frame, (4, 4), (w - 4, h - 4), flash_color, 6, cv2.LINE_AA)

            draw_hotkeys(frame, w, h, [
                ("SPACE", "Next Round", ui.GOLD), ("T", "Retrain", ui.CYAN_P1),
                ("R", "Reset", ui.WHITE), ("Q", "Quit", ui.RED),
            ])

        elif state == STATE_GAME_OVER:
            cx, cy = w // 2, (HEADER_H + h - FOOTER_H) // 2
            winner_name = "PLAYER 1" if score1 > score2 else "PLAYER 2"
            winner_color = ui.CYAN_P1 if score1 > score2 else ui.MAGENTA_P2
            ui.rounded_rect(frame, (cx - 260, cy - 110), (cx + 260, cy + 110), ui.PANEL, radius=26, alpha=0.9)
            ui.rounded_rect(frame, (cx - 260, cy - 110), (cx + 260, cy + 110), ui.GOLD, radius=26, thickness=3)
            ui.icon_trophy(frame, (cx, cy - 55), 34, ui.GOLD)
            ui.centered_text(frame, f"{winner_name} WINS THE MATCH!", cx, cy + 15, 0.9, winner_color, 2)
            ui.centered_text(frame, f"Final Score  {score1} - {score2}", cx, cy + 55, 0.6, ui.WHITE, 1)
            cv2.rectangle(frame, (4, 4), (w - 4, h - 4), ui.GOLD, 8, cv2.LINE_AA)

            draw_hotkeys(frame, w, h, [
                ("SPACE", "New Match", ui.GOLD), ("T", "Retrain", ui.CYAN_P1), ("Q", "Quit", ui.RED),
            ])

        cv2.imshow("Rock Paper Scissors", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('t'):
            train_dataset = {label: [] for label in LABELS}
            state_before_train = STATE_IDLE
            state = STATE_TRAIN
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord(' ') and state in (STATE_IDLE, STATE_RESULT):
            state = STATE_COUNTDOWN
            countdown_start = time.time()
        elif key == ord(' ') and state == STATE_GAME_OVER:
            score1, score2, round_no = 0, 0, 1
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
