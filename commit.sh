dvc add $(cat dvc_files.txt)
git add *
git commit -m "$1"
git push
dvc push
