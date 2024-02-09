#!/bin/bash
export CUDA_VISIBLE_DEVICES="0" 
python generation_rec_agents.py    \
 --task movie     --backend gpt-3.5-turbo-16k    \
 --promptpath cot_movie_upper     --evaluate     \
 --random     --task_split test     \
 --temperature 0     --task_end_index 1   \
 --env movie  --env_threshold 3  \
 --Max_Iteration 50 --agent_name agent_revise --Max_Reflections 2 --batch_size 5