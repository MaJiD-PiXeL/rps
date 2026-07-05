"""
hand_utils.py
=============
لایه‌ی مشترک تشخیص دست بین اسکریپت آموزش (train_gestures.py) و بازی
(rps_game.py). به‌جای تشخیص رنگ پوست، از MediaPipe Hand Landmarker
(API جدید Tasks) برای پیدا کردن 21 نقطه‌ی اسکلت دست استفاده می‌شود؛
این روش مستقل از نور و رنگ پوست است و دقت را به‌شدت بالا می‌برد.

توجه: نسخه‌های جدید mediapipe (>=0.10.x) دیگر API قدیمی
`mediapipe.solutions.hands` را ندارند و باید از API جدید Tasks استفاده
شود که به یک فایل مدل (hand_landmarker.task) نیاز دارد. این فایل فقط
بار اول به‌صورت خودکار دانلود می‌شود (حدود 8 مگابایت) و کنار همین
اسکریپت‌ها ذخیره می‌شود تا دفعات بعد نیازی به اینترنت نباشد.
"""

import os
import time
import urllib.request

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        HandLandmarksConnections,
        RunningMode,
    )
except ImportError as e:
    raise ImportError(
        "کتابخانه mediapipe نصب نیست یا نسخه‌ی آن ناقص است. اجرا کن:\n"
        "    pip install --upgrade mediapipe"
    ) from e

HAND_CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_FILENAME = "hand_landmarker.task"


def _ensure_model_downloaded():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODEL_FILENAME)
    if not os.path.exists(model_path):
        print("در حال دانلود مدل تشخیص دست (فقط یک‌بار، حدود ۸ مگابایت)...")
        try:
            urllib.request.urlretrieve(_MODEL_URL, model_path)
            print("دانلود مدل با موفقیت انجام شد.")
        except Exception as e:
            raise RuntimeError(
                "دانلود خودکار مدل تشخیص دست ناموفق بود. اینترنت را چک کن، یا این "
                f"فایل را دستی دانلود کن و کنار اسکریپت‌ها با همین نام بگذار:\n"
                f"{_MODEL_URL}\nنام فایل باید دقیقاً '{_MODEL_FILENAME}' باشد."
            ) from e
    return model_path


def extract_features(points_xyz: np.ndarray) -> np.ndarray:
    """
    ورودی: آرایه (21, 3) مختصات نرمال‌شده‌ی MediaPipe (x, y, z بین 0 و 1).
    خروجی: بردار ویژگی 63 بعدی، نسبت به مچ دست جابه‌جا و بر اساس اندازه‌ی
    کف دست مقیاس‌شده (Scale & Translation Invariant) تا فاصله‌ی دست از
    دوربین و موقعیتش در قاب، روی تشخیص تأثیر نگذارد.
    """
    wrist = points_xyz[0].copy()
    translated = points_xyz - wrist
    scale = np.linalg.norm(points_xyz[9] - wrist)  # فاصله مچ تا قاعده انگشت وسط
    if scale < 1e-6:
        scale = 1e-6
    return (translated / scale).flatten().astype(np.float32)


class HandTracker:
    """رَپِر ساده روی MediaPipe HandLandmarker (Tasks API) برای تشخیص تا 2 دست هم‌زمان."""

    def __init__(self, max_hands=2, det_conf=0.6, presence_conf=0.5, track_conf=0.5):
        model_path = _ensure_model_downloaded()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=presence_conf,
            min_tracking_confidence=track_conf,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def process(self, frame_bgr):
        """
        خروجی: لیستی از دیکشنری برای هر دست پیدا‌شده:
            {
              "landmarks_px": [(x, y), ... 21 نقطه به پیکسل],
              "features": np.ndarray(63,),
              "center_x": float  (میانگین x به پیکسل - برای تشخیص چپ/راست بودن)
            }
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        hands_out = []
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                pts = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                    dtype=np.float32,
                )
                px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                hands_out.append(
                    {
                        "landmarks_px": px,
                        "features": extract_features(pts),
                        "center_x": float(np.mean(pts[:, 0]) * w),
                    }
                )
        return hands_out

    def close(self):
        self.landmarker.close()


def draw_hand_skeleton(img, landmarks_px, line_color, point_color=None, thickness=2):
    """اسکلت دست را با رنگ اختصاصیِ بازیکن روی تصویر می‌کشد."""
    point_color = point_color or line_color
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, landmarks_px[a], landmarks_px[b], line_color, thickness, cv2.LINE_AA)
    for p in landmarks_px:
        cv2.circle(img, p, 4, point_color, -1, cv2.LINE_AA)
    # نقطه‌ی مچ را بزرگ‌تر مشخص می‌کنیم
    cv2.circle(img, landmarks_px[0], 7, point_color, 2, cv2.LINE_AA)
