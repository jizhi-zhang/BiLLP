#!/bin/bash
export CUDA_VISIBLE_DEVICES="6" 
python generation_rec_agents.py    \
 --task steam     --backend llama     \
 --promptpath cot_movie_upper     --evaluate     \
 --random     --task_split train     \
 --temperature 0.5   --task_end_index 2   \
 --env steam  --env_threshold 50  --env_window_length 4 \
 --Max_Iteration 50 --agent_name agent_reflection --Max_Reflections 0 --batch_size 2