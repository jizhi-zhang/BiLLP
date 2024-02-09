
## Overview
- Define tasks in `tasks/`
- Collect data & run experiments via `generation_rec_agents.py` 
- Results will be saved in `trajs_agent/`


## Data & Prompts
- Data to generate training data and run experiments in `data/`. 
- Download data from [steam_test.npy](https://ufile.io/4pva322t), [steam_train.npy](https://ufile.io/uhtfwv5d), [test_distance_mat.pickle](https://ufile.io/frft6hr2), [train_distance_mat.pickle](https://ufile.io/jrshloa1) to `/data/steam/`

## Setup

Set up OpenAI API key and store in environment variable  (see [here](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety))

```
export OPENAI_API_KEY=<YOUR_KEY>
```
 
Create virtual env, for example with conda

```
conda create -n BiLLP python=3.9
conda activate BiLLP
```

Install dependencies

```
pip install -r requirements.txt
```

## Run Demo

#### Data Generation

Example:

train and test
```
source run_steam.sh
source run_steam_test.sh
```

```
source run_amazon.sh
source run_amazon_test.sh
```


## References
Our  code is based on [FireAct](https://github.com/anchen1011/FireAct)