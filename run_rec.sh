#!/bin/bash
export CUDA_VISIBLE_DEVICES="1" 
python generation_rec.py    \
 --task movie     --backend gpt-3.5-turbo-16k     \
 --promptpath cot_movie_2     --evaluate     \
 --random     --task_split train     \
 --temperature 0.5    --task_end_index 10   \
 --env movie  --env_threshold 30  --env_window_length 4\
 --Max_Iteration 50 