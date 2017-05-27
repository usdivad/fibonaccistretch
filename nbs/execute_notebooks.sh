# Script to execute Jupyter notebooks and save output

# Args
# commit_msg=$1

# Navigate to proper directory
cd ~/Documents/tech/fibonacciswing/nbs

# Create temporary copies of notebooks
cp fibonaccistretch.ipynb tmp_fibonaccistretch_executed.ipynb
cp export_with_figs.ipynb tmp_export_with_figs_executed.ipynb
cp link_ipd_audio.ipynb tmp_link_ipd_audio_executed.ipynb

# Remove existing figures
rm -rf ../data/figs
mkdir ../data/figs

# Run the notebooks
jupyter nbconvert --to notebook --execute tmp_fibonaccistretch_executed.ipynb
jupyter nbconvert --to notebook --execute tmp_export_with_figs_executed.ipynb
jupyter nbconvert --to notebook --execute tmp_link_ipd_audio_executed.ipynb

# Remove all temporary files
rm tmp_*.ipynb
# rm tmp_fibonaccistretch_executed.ipynb
# rm tmp_export_with_figs_executed.ipynb
# rm tmp_link_ipd_audio_executed.ipynb

git add -u
git add ../data/figs
# git commit -m "$1"