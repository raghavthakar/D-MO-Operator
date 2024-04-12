#!/bin/zsh
current_date_time=$(date +"%Y-%m-%d %T")

filename="${current_date_time}.txt"

touch "data/$filename"

cat ../config/config.yaml >> data/$filename 

echo "
" >> data/$filename

./../build/MOD "../config/config.yaml" "../experiments/data/$filename" >> data/$filename