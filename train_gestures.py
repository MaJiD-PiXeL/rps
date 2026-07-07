"""
train_gestures.py
==================
با این اسکریپت خودت مدل تشخیص حرکت دست را آموزش می‌دهی:
دستت را جلوی دوربین به حالت سنگ/کاغذ/قیچی می‌گیری، کلید مربوطه را می‌زنی
تا آن حالت به‌عنوان یک نمونه ذخیره شود؛ بعد از جمع‌آوری چند ده نمونه از
هر حالت (بهتر است با زاویه و فاصله‌ی کمی متفاوت)، کلید آموزش را می‌زنی
تا یک مدل RandomForest روی داده‌های خودِ دست تو ساخته و ذخیره شود.
این مدل بعداً توسط rps_game.py استفاده می‌شود.

اجرا:
    pip install opencv-python mediapipe scikit-learn joblib numpy
    python train_gestures.py

کنترل‌ها:
    1        -> ثبت نمونه‌ی فعلی به‌عنوان ROCK (سنگ)
    2        -> ثبت نمونه‌ی فعلی به‌عنوان PAPER (کاغذ)
    3        -> ثبت نمونه‌ی فعلی به‌عنوان SCISSORS (قیچی)
    BACKSPACE -> حذف آخرین نمونه‌ی ثبت‌شده (در صورت اشتباه)
    T        -> آموزش مدل با نمونه‌های فعلی و ذخیره روی دیسک
    Q / ESC  -> خروج بدون آموزش
"""

import os
import time
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from hand_utils import HandTracker, draw_hand_skeleton, extract_features, augment_landmarks
import ui_kit as ui

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_model.pkl")
LABELS = ["ROCK", "PAPER", "SCISSORS"]
KEY_TO_LABEL = {ord('1'): "ROCK", ord('2'): "PAPER", ord('3'): "SCISSORS"}
MIN_SAMPLES_PER_CLASS = 15
RECOMMENDED_SAMPLES = 40


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("خطا: وب‌کم پیدا نشد.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker(max_hands=1)
    dataset = {label: [] for label in LABELS}
    last_captured_label = None
    last_captured_time = 0
    flash_until = 0
    status_msg = "دستت رو جلوی دوربین بگیر و با کلیدهای 1 2 3 نمونه ثبت کن"

    cv2.namedWindow("Train Gestures", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        hands = tracker.process(frame)
        current_hand = hands[0] if hands else None

        if current_hand:
            draw_hand_skeleton(frame, current_hand["landmarks_px"], ui.GOLD, ui.WHITE)

        # ---------------- هدر ----------------
        ui.rounded_rect(frame, (0, 0), (w, 90), ui.BG_DARK, radius=0, alpha=0.65)
        ui.centered_text(frame, "GESTURE TRAINER", w // 2, 40, 1.0, ui.GOLD, 2)
        ui.centered_text(frame, "دست خودت را نشون بده و کلید 1/2/3 را بزن", w // 2, 72, 0.55, ui.WHITE, 1)

        # ---------------- پنل شمارش نمونه‌ها ----------------
        panel_y = h - 130
        ui.rounded_rect(frame, (20, panel_y), (w - 20, h - 20), ui.PANEL, radius=18, alpha=0.75)
        col_w = (w - 40) // 3
        colors = {"ROCK": ui.CYAN_P1, "PAPER": ui.GOLD, "SCISSORS": ui.MAGENTA_P2}
        keys_hint = {"ROCK": "[1]", "PAPER": "[2]", "SCISSORS": "[3]"}
        for i, label in enumerate(LABELS):
            cx = 20 + col_w * i + col_w // 2
            count = len(dataset[label])
            ratio = min(count / RECOMMENDED_SAMPLES, 1.0)
            ui.draw_gesture_icon(frame, label, (cx, panel_y + 35), 22, colors[label])
            ui.centered_text(frame, f"{keys_hint[label]} {label}", cx, panel_y + 75, 0.6, colors[label], 1)
            ui.progress_bar(frame, (cx - 70, panel_y + 85), (cx + 70, panel_y + 98), ratio, colors[label])
            ui.centered_text(frame, f"{count} sample", cx, panel_y + 118, 0.5, ui.WHITE, 1)

        # ---------------- وضعیت لحظه‌ای ----------------
        if current_hand is None:
            hint_color = ui.RED
            hint_text = "دستی دیده نمی‌شود..."
        else:
            hint_color = ui.GREEN
            hint_text = "دست شناسایی شد - آماده‌ی ثبت نمونه"
        ui.centered_text(frame, hint_text, w // 2, 115, 0.6, hint_color, 1)

        total = sum(len(v) for v in dataset.values())
        min_ok = all(len(v) >= MIN_SAMPLES_PER_CLASS for v in dataset.values())
        train_hint_color = ui.GREEN if min_ok else ui.GRAY
        train_msg = "T: آموزش مدل" if min_ok else f"T: آموزش (حداقل {MIN_SAMPLES_PER_CLASS} نمونه از هر کلاس لازم است)"
        ui.centered_text(frame, train_msg, w // 2, panel_y - 15, 0.55, train_hint_color, 1)

        if time.time() < flash_until:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), ui.GREEN, 10)
            ui.centered_text(frame, f"{last_captured_label} sample saved!", w // 2, h // 2, 0.9, ui.GREEN, 2)

        cv2.imshow("Train Gestures", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key in KEY_TO_LABEL and current_hand is not None:
            label = KEY_TO_LABEL[key]
            dataset[label].append(current_hand["landmarks_norm"])
            last_captured_label = label
            last_captured_time = time.time()
            flash_until = time.time() + 0.35

        elif key == 8:  # Backspace -> حذف آخرین نمونه ثبت‌شده
            if last_captured_label and dataset[last_captured_label]:
                dataset[last_captured_label].pop()
                status_msg = f"آخرین نمونه‌ی {last_captured_label} حذف شد"

        elif key == ord('t'):
            if not min_ok:
                status_msg = "هنوز نمونه‌ی کافی جمع نشده."
                continue

            # هر نمونه‌ی خام را با augment_landmarks به چند نسخه (آینه‌ای + نویزی)
            # تبدیل می‌کنیم تا مدل در برابر تغییر زاویه/دست چپ‌وراست مقاوم‌تر شود.
            X, y = [], []
            for label, raw_samples in dataset.items():
                for raw_pts in raw_samples:
                    for variant in augment_landmarks(raw_pts):
                        X.append(extract_features(variant))
                        y.append(label)
            X = np.array(X)
            y = np.array(y)

            if len(set(y)) < 3:
                status_msg = "باید از هر سه حالت نمونه داشته باشی."
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            clf = RandomForestClassifier(n_estimators=300, max_depth=16, random_state=42)
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test)) if len(X_test) > 0 else float("nan")

            # مدل نهایی روی کل داده (برای بیشترین دقت در بازی)
            final_clf = RandomForestClassifier(n_estimators=300, max_depth=16, random_state=42)
            final_clf.fit(X, y)
            saved_labels = list(final_clf.classes_)  # ترتیب واقعی کلاس‌ها از خودِ مدل
            joblib.dump({"model": final_clf, "labels": saved_labels}, MODEL_PATH)

            print(f"مدل ذخیره شد در: {MODEL_PATH}")
            print(f"دقت روی داده‌ی تست (validation): {acc * 100:.1f}%")
            print(f"تعداد نمونه‌های خام: { {k: len(v) for k, v in dataset.items()} }")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
