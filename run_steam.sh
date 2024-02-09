#!/bin/bash
export CUDA_VISIBLE_DEVICES="0" 
python generation_rec_agents.py    \
 --task steam     --backend gpt-3.5-turbo-16k     \
 --promptpath cot_movie_upper     --evaluate     \
 --random     --task_split train     \
 --temperature 0.5   --task_end_index 100   \
 --env steam  --env_threshold 50  --env_window_length 4 \
 --Max_Iteration 50 --agent_name agent_a2c --Max_Reflections 2 --batch_size 10