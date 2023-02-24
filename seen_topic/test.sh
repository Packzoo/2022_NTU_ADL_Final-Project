# "${1}" : --test_data ../data/test_seen_group.csv
# "${2}" : --output_data ./output/result_group_seen.csv
python3 test.py \
    --input_dir ../input/hahow/ \
    --save_dir ../output/seen_topic/ \
    --train_data ../input/hahow/train_group.csv \
    --valid_data ../input/hahow/val_seen_group.csv \
    --test_data "${1}" \
    --output_data "${2}" \
    --test
