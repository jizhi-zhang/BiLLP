#!/bin/bash
export CUDA_VISIBLE_DEVICES="2" 
python generation_rec_agents.py    \
 --task amazon     --backend gpt-4-32k     \
 --promptpath cot_movie_upper     --evaluate     \
 --random     --task_split test     \
 --temperature 0.5   --task_end_index 10   \
 --env amazon  --env_threshold 15  --env_window_length 4 \
 --Max_Iteration 50 --agent_name agent_a2c --Max_Reflections 0  --batch_size 10 \
 --input_file_name amazon_train_0_100_gpt-3.5-turbo-16k_0.5_2024-01-17-21-28-32

 
 