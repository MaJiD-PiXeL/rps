"""
audio.py
========
افکت‌های صوتی ساده‌ی بازی (تیک شمارش معکوس، شروع دور، برد، باخت، تساوی)
بدون نیاز به هیچ فایل صوتی خارجی یا کتابخانه‌ی اضافه؛ فقط با بیپ‌های
سینتی‌سایزشده از طریق ماژول استاندارد winsound (فقط ویندوز).

روی سیستم‌عامل‌های غیر از ویندوز، این افکت‌ها بی‌صدا نادیده گرفته
می‌شوند (بدون خطا) تا برنامه همچنان اجرا شود.
"""

import threading

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

_enabled = True


def set_enabled(value: bool):
    global _enabled
    _enabled = value


def is_enabled() -> bool:
    return _enabled


def _run_async(fn):
    if not _HAS_WINSOUND or not _enabled:
        return
    threading.Thread(target=fn, daemon=True).start()


def play_tick():
    """صدای کوتاه تیک برای هر عدد شمارش معکوس (۳، ۲، ۱)."""
    _run_async(lambda: winsound.Beep(440, 110))


def play_go():
    """صدای شروع لحظه‌ی ثبت حرکت (وقتی شمارش به صفر می‌رسد)."""
    _run_async(lambda: winsound.Beep(880, 160))


def play_win():
    """آرپژ کوتاه صعودی برای اعلام برد یک راند."""
    def seq():
        for f in (660, 880, 1046):
            winsound.Beep(f, 110)
    _run_async(seq)


def play_tie():
    _run_async(lambda: winsound.Beep(320, 220))


def play_lose():
    _run_async(lambda: winsound.Beep(220, 260))


def play_match_win():
    """فانفار بلندتر برای پیروزی در کل مسابقه (پایان بازی)."""
    def seq():
        for f in (523, 659, 784, 1046):
            winsound.Beep(f, 140)
    _run_async(seq)
