#!/bin/bash

# Function to run Python script and capture time and output
run_script() {
    start_time=$(date +%s.%N)
    output=$(python txt2image.py --n_images "$1" --steps 50 --cache_interval 5 --cache_layer_id 2 --model sdxl -q -v --output sd_img="$1"_interval=5_layer=2.png "A painting of a vase on a wooden table, dark background, still life.")
    end_time=$(date +%s.%N)
    execution_time=$(echo "($end_time - $start_time) * 1000" | bc)
    echo "Execution time for $1: $execution_time milliseconds" >> sdxl_image-test_output.txt
    echo "Output of $1:" >> sdxl_image-test_output.txt
    echo "$output" >> sdxl_image-test_output.txt
    echo "==========================================" >> sdxl_image-test_output.txt
}

# Run each Python script and capture output
run_script 1
run_script 2
run_script 4
run_script 10
