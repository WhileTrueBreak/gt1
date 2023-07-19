dvc add $(cat dvc_files.txt)
git add *
echo $1
git commit -m "$1"
git push
dvc push
