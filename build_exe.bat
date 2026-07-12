@echo off
REM ============================================================
REM  build_exe.bat
REM  ساخت یک فایل اجرایی مستقل (.exe) از بازی، برای اجرا روی
REM  ویندوزهای دیگر بدون نیاز به نصب پایتون یا کتابخانه‌ها.
REM
REM  قبل از اجرای این اسکریپت بهتر است یک‌بار "python rps_game.py"
REM  را عادی اجرا کرده باشی و مدل را آموزش داده باشی، تا exe نهایی
REM  از همان اول با مدل و بدون نیاز به اینترنت کار کند. اگر این کار
REM  را نکنی، هم اشکالی ندارد: خودِ exe در اولین اجرا فایل تشخیص
REM  دست را دانلود می‌کند و راهنمای تمرین درون‌برنامه را نشان می‌دهد.
REM ============================================================

echo در حال نصب PyInstaller (اگر از قبل نصب نباشد)...
pip install --quiet pyinstaller

echo.
echo در حال ساخت فایل exe... این ممکن است چند دقیقه طول بکشد.
echo.

pyinstaller --onefile --noconsole --name RockPaperScissors rps_game.py

echo.
echo کپی کردن فایل‌های مدل (در صورت وجود) کنار exe نهایی...
REM توجه: این دو فایل عمداً به‌جای bundle شدن داخل exe، کنارش کپی
REM می‌شوند؛ چون exe در زمان اجرا دنبال آن‌ها در کنار خودش می‌گردد
REM (نه در پوشه‌ی موقتی که PyInstaller هنگام اجرای onefile می‌سازد).
if exist hand_landmarker.task copy /Y hand_landmarker.task dist\ >nul
if exist gesture_model.pkl copy /Y gesture_model.pkl dist\ >nul

echo.
echo تمام شد! همه‌چیز داخل پوشه‌ی dist است:
echo    dist\RockPaperScissors.exe
if exist dist\gesture_model.pkl (
    echo    dist\gesture_model.pkl   (مدل آموزش‌دیده‌ی تو - همراه است)
) else (
    echo    (مدلی کنارش نبود؛ کاربر باید در اولین اجرا با کلید T تمرین بدهد)
)
echo.
echo نکته: کل پوشه‌ی dist را برای دوستت بفرست (نه فقط فایل exe تنها)
echo تا فایل‌های کنارش هم همراهش باشند. نیازی به نصب پایتون یا هیچ
echo کتابخانه‌ای روی سیستم او نیست.
pause
