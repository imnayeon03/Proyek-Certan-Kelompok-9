@echo off
title BATTLESHIP - Perang Kapal (Kelompok 9)
echo Memulai BATTLESHIP Pygame...
"C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe" battleship_pygame.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Game gagal dijalankan. Pastikan pygame sudah terinstal:
    echo   python -m pip install pygame numpy
    pause
)
