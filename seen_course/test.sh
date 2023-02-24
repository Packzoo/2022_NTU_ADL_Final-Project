#${1} : input path
#${2} : output path

python3 Test.py \
    --input_dir ../input/hahow \
    --save_dir ../output/seen_course/epoch2 \
    --test_data ${1} \
    --output_data ${2} \
    --test  