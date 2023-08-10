@echo off

cd ..
cd ..
if exist raylib.dll (code\raylib\remedybg.exe code\raylib\main.rdbg) else (main.exe)
