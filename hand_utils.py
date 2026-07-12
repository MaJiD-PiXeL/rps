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
import sys
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


def get_base_dir():
    """
    پوشه‌ای که فایل‌های کنارِ برنامه (مدل، تنظیمات) باید در آن ذخیره/خوانده شوند.
    وقتی برنامه به exe تبدیل شده باشد (PyInstaller)، فایل‌های bundle‌شده در
    یک پوشه‌ی موقت استخراج می‌شوند که بین اجراها پاک می‌شود؛ برای همین در آن
    حالت باید کنار خودِ فایل exe (نه پوشه‌ی موقت) ذخیره کنیم تا ماندگار بماند.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

_MODEL_URLS = [
    # آدرس پایدار و نسخه‌دار (همانی که در نمونه‌های رسمی گوگل استفاده می‌شود)
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    # آدرس جایگزین (alias به آخرین نسخه) در صورتی که مورد بالا در دسترس نبود
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
]
_MODEL_FILENAME = "hand_landmarker.task"


def _download_file(url, dest_path):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RPS-Game/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response, open(dest_path, "wb") as out_file:
        total = response.getheader("Content-Length")
        total = int(total) if total else None
        downloaded = 0
        while True:
            chunk = response.read(262144)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  دانلود: {downloaded * 100 / total:5.1f}%", end="", flush=True)
        print()


def _ensure_model_downloaded():
    model_path = os.path.join(get_base_dir(), _MODEL_FILENAME)
    if os.path.exists(model_path):
        return model_path

    print("در حال دانلود مدل تشخیص دست (فقط یک‌بار، حدود ۸ مگابایت)...")
    tmp_path = model_path + ".part"
    last_error = None
    for url in _MODEL_URLS:
        try:
            _download_file(url, tmp_path)
            os.replace(tmp_path, model_path)
            print("دانلود مدل با موفقیت انجام شد.")
            return model_path
        except Exception as e:
            last_error = e
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"  تلاش با یک آدرس ناموفق بود ({type(e).__name__})، امتحان آدرس بعدی...")

    raise RuntimeError(
        "دانلود خودکار مدل تشخیص دست ناموفق بود (همه‌ی آدرس‌ها رد شدند). "
        "اتصال اینترنت یا فایروال/آنتی‌ویروس را چک کن، یا فایل را دستی از "
        "یکی از این آدرس‌ها با مرورگر دانلود کن و کنار اسکریپت‌ها با نام دقیق "
        f"'{_MODEL_FILENAME}' ذخیره کن:\n" + "\n".join(_MODEL_URLS)
    ) from last_error


def extract_features(points_xyz: np.ndarray) -> np.ndarray:
    """
    ورودی: آرایه (21, 3) مختصات نرمال‌شده‌ی MediaPipe (x, y, z بین 0 و 1).
    خروجی: بردار ویژگی 63 بعدی که نسبت به سه چیز ثابت (Invariant) است:
      1) جابه‌جایی دست در قاب (نسبت به مچ جابه‌جا می‌شود)
      2) فاصله‌ی دست از دوربین / بزرگی دست (بر اساس اندازه‌ی کف دست مقیاس می‌شود)
      3) زاویه‌ی چرخش دست در تصویر (چرخانده می‌شود تا همیشه «رو به بالا» باشد)

    مورد سوم قبلاً نبود و باعث می‌شد اگر زاویه‌ی نگه‌داشتن دست موقع بازی
    کمی با زاویه‌ی موقع تمرین فرق کند، مدل اشتباه تشخیص بدهد.
    """
    pts = points_xyz.copy()
    wrist = pts[0].copy()
    translated = pts - wrist

    mid_mcp = translated[9]  # قاعده‌ی انگشت وسط - محور مرجع جهت دست
    current_angle = np.arctan2(mid_mcp[1], mid_mcp[0])
    target_angle = np.arctan2(-1.0, 0.0)  # می‌خواهیم این بردار همیشه رو به بالا باشد
    delta = target_angle - current_angle
    cos_d, sin_d = np.cos(delta), np.sin(delta)

    x, y = translated[:, 0].copy(), translated[:, 1].copy()
    rotated = translated.copy()
    rotated[:, 0] = x * cos_d - y * sin_d
    rotated[:, 1] = x * sin_d + y * cos_d

    scale = np.linalg.norm(rotated[9, :2])
    if scale < 1e-6:
        scale = 1e-6
    rotated /= scale

    return rotated.flatten().astype(np.float32)


def augment_landmarks(pts: np.ndarray, num_noise=3, noise_sigma=0.01):
    """
    از یک نمونه‌ی خام (21,3) چند نسخه‌ی متفاوت می‌سازد تا مدل با تنوع بیشتری
    آموزش ببیند بدون این‌که کاربر مجبور باشد نمونه‌های بیشتری دستی بگیرد:
      - نسخه‌ی اصلی
      - نسخه‌ی آینه‌ای (برای این‌که مدل به دست چپ/راست بازیکن دوم هم حساس نباشد)
      - چند نسخه با کمی نویز تصادفی روی هرکدام (برای مقاومت در برابر لرزش/نویز دوربین)
    خروجی: لیستی از آرایه‌های (21,3) که باید بعداً با extract_features تبدیل شوند.
    """
    variants = [pts]

    mirrored = pts.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    variants.append(mirrored)

    rng = np.random.default_rng()
    base_variants = list(variants)
    for v in base_variants:
        for _ in range(num_noise):
            noisy = v + rng.normal(0, noise_sigma, size=v.shape).astype(np.float32)
            variants.append(noisy)

    return variants


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
                        "landmarks_norm": pts,
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
