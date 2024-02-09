#!/bin/bash
export CUDA_VISIBLE_DEVICES="2" 
python generation_rec_agents.py    \
 --task amazon     --backend gpt-3.5-turbo-16k     \
 --promptpath cot_movie_upper     --evaluate     \
 --random     --task_split train     \
 --temperature 0.5   --task_end_index 100   \
 --env amazon  --env_threshold 15  --env_window_length 4 \
 --Max_Iteration 50 --agent_name agent_a2c --Max_Reflections 2  --batch_size 10 \

 