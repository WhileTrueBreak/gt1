@echo off
set arg1=%1
for /f usebackq %%F in (`type dvc_files.txt`) do dvc add %%F
git add *
git commit -m %arg1%
git push
dvc push