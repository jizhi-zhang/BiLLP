import re, string, os
from typing import List, Union, Literal
from enum import Enum
import tiktoken
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore.base import Docstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from Agents.llm import AnyOpenAILLM, OpenAILLM
from Agents.prompts import reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, react_reflect_retrival_agent_prompt, critic_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from Agents.prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt, COT_INSTRUCTION, COT_REFLECT_INSTRUCTION
from Agents.fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT
from Agents.agent_base import ReactAgent, parse_action, format_step, truncate_scratchpad
from Agents.agent_reflexion import ReactReflectAgent, ReflexionStrategy
import random
from collections import defaultdict
import json
import numpy as np
import openai
import time

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_info(infos, outfilename):
    if 'trajs' in infos:
        traj_file_name = f"trajs_agent/{outfilename}.json"
        with open(traj_file_name, "w") as fout:
            json.dump(infos['trajs'], fout, indent=2)
    
    if 'reflections' in infos:
        reflection_file_name = f'reflections/{outfilename}.txt'
        with open(reflection_file_name, 'w') as file:
            for item in infos['reflections']:
                file.write(str(item) + '\n')
    
    if 'Q_table' in infos:
        memory_file_name = f'memory/{outfilename}.json'
        with open(memory_file_name, "w") as fout:
            json.dump(infos['Q_table'], fout, indent=2, cls=NpEncoder)
            
    if 'actor_memory' in infos:
        memory_file_name = f'memory/{outfilename}.json'
        with open(memory_file_name, "w") as fout:
            json.dump(infos['actor_memory'], fout, indent=2, cls=NpEncoder)
    
    if 'critic_memory' in infos:
        critic_memory_file_name = f'critic_memory/{outfilename}.json'
        with open(critic_memory_file_name, "w") as fout:
            json.dump(infos['critic_memory'], fout, indent=2, cls=NpEncoder)

class ReactA2CAgent(ReactReflectAgent):
    def __init__(self,
                 task,
                 idxs: list, 
                 args, 
                 rec_env,
                 grounding_model,
                 max_steps: int = 30,
                 agent_prompt: PromptTemplate = react_reflect_retrival_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 react_llm: AnyOpenAILLM = AnyOpenAILLM(
                                             temperature=0,
                                             max_tokens=8000,
                                             model_name="gpt-3.5-turbo-16k",
                                             model_kwargs={"stop": "\n"},
                                             openai_api_key=os.environ['OPENAI_API_KEY'],
                                             openai_api_base = os.environ['OPENAI_API_BASE']),
                 reflect_llm: AnyOpenAILLM = AnyOpenAILLM(
                                               temperature=0,
                                               max_tokens=8000,
                                               model_name="gpt-3.5-turbo-16k",
                                               openai_api_key=os.environ['OPENAI_API_KEY'],
                                               openai_api_base = os.environ['OPENAI_API_BASE']),
                 critic_llm: AnyOpenAILLM = AnyOpenAILLM(
                                               temperature=0,
                                               max_tokens=8000,
                                               model_name="gpt-3.5-turbo-16k",
                                               openai_api_key=os.environ['OPENAI_API_KEY'],
                                               openai_api_base = os.environ['OPENAI_API_BASE']),
                 reflections_memory = None,
                 actor_memory = None,
                 critic_memory = None,
                 ) -> None:
        
        super().__init__(task, idxs, args, rec_env, grounding_model, max_steps, agent_prompt, react_llm, reflect_llm)
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.critic_prompt = critic_prompt
        self.reflect_examples = REFLECTIONS
        self.critic_llm = critic_llm
        self.reflections_str: dict = {}
        
        
        if reflections_memory == None:
            self.reflections: list = []
            self.faiss_reflections = None
        else:
            self.reflections = reflections_memory
            self._update_reflections_lib()
        
        if actor_memory == None:
            self.actor_memory: dict =defaultdict(dict) 
            self.faiss_actor_memory = None
        else:
            self.actor_memory = actor_memory
            embeddings = OpenAIEmbeddings()
            self.faiss_actor_memory = FAISS.from_texts(self.actor_memory.keys(), embeddings)

        if critic_memory == None:
            self.critic_memory: dict =defaultdict(dict) 
            self.faiss_critic_memory = None
        else:
            self.critic_memory = critic_memory
            embeddings = OpenAIEmbeddings()
            self.faiss_critic_memory = FAISS.from_texts(self.critic_memory.keys(), embeddings)
        
        self.infos = {}
        self.final_infos = {}
        
        self.batch_size = args.batch_size
        self.enc = tiktoken.encoding_for_model("text-davinci-003")
    
    def run(self, reset = True, reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION, outfilename='') -> None:
        
        for i in range(0, len(self.idxs), self.batch_size):
            temp_idxs = self.idxs[i: i+self.batch_size]
            print(f'temp_idxs:{temp_idxs}')
            
            self.get_reflect_str(reflect_strategy, temp_idxs)
            
            self.single_run(temp_idxs, reset)

            self.reflect(reflect_strategy, temp_idxs)
            self._update_reflections_lib()
            self._update_memory(temp_idxs)
        
            self._build_info(temp_idxs)
        
            self.final_infos['trajs'] = self.infos
            self.final_infos['reflections'] = self.reflections
            self.final_infos['actor_memory'] = self.actor_memory
            self.final_infos['critic_memory'] = self.critic_memory
            save_info(self.final_infos, outfilename)
            
        return self.final_infos
    
    def step(self, idxs) -> None:
        # Think
        for id in idxs:
            self.scratchpad[id] += f'\nThought {self.step_n}:'
        prompts = self.prompt_agent(idxs)
        for i, id in enumerate(idxs):
            self.scratchpad[id] += ' ' + prompts[i]
            
        if self.tool_use ==True:
        # print(self.scratchpad.split('\n')[-1])
            random_type = []
            q_prompt = {}
            for i, id in enumerate(idxs):
                hist = self.env.get_hist_list(self.argument_lists[id])
                random_type.append(random.sample([x for x in self.GENRE if x not in hist], 2))
                self.scratchpad[id] += f'(Please recommend {random_type[i][0]} and {random_type[i][1]} items to help users explore their interests)'
            
                if self.faiss_actor_memory!=None:
                    q_prompt[id] = self._get_actor_memory(self.task.get_history_actions(id), self.argument_lists[id])
                    self.scratchpad[id] += q_prompt[id]
                    print(q_prompt[id])
                
        # Act
        for id in idxs:
            self.scratchpad[id] += f'\nAction {self.step_n}:'
        for _ in range(5):
            try:
                action = self.prompt_agent(idxs)
                action_types, arguments = parse_action(action)
                print(f'a:{action}')
                break
            except:
                print('b')
                continue
            
        for i, id in enumerate(idxs):
            self.scratchpad[id] = self.scratchpad[id].replace(f'(Please recommend {random_type[i][0]} and {random_type[i][1]} items to help users explore their interests)', '')
            if self.faiss_actor_memory!=None:
                self.scratchpad[id] = self.scratchpad[id].replace(q_prompt[id], '')
        
             
        
        for i, id in enumerate(idxs):
            
            self.scratchpad[id] += ' ' + action[i]
            
            if action_types[i] == 'recommend':
                old_film = arguments[i]
                argument_candidate = self.grounding_model.get_topk_near_item([old_film], self.args.Max_Iteration)[0]
                arguments[i] = argument_candidate[0]
                self.argument_lists[id].append(arguments[i])
                if old_film != arguments[i]:
                    self.scratchpad[id] += f'\nObservation {self.step_n}: [{old_film}] can not be recommened, instead, recommend[{arguments[i]}]'
                

        # Observe
        for i, id in enumerate(idxs):
            self.scratchpad[id] +=  f'\nObservation {self.step_n}: '
        
            if action_types[i] == 'recommend':
                reward = self.env.get_reward(self.userids[id], arguments[i])
                if self.env.whether_to_leave(self.userids[id], arguments[i], self.argument_lists[id]):
                    self.scratchpad[id] += f"Episode finished, User Stop, reward=-1000.000"
                    self.reward_lists[id].append(-1000)
                    self.finished[id] = True
                else:
                    self.scratchpad[id] += f"Episode continue, reward={reward}"
                    self.reward_lists[id].append(reward)

            else:
                self.scratchpad[id] += 'Invalid Action. Valid Actions are recommend[item].'
                self.finished[id] = True
        
        # update actor memory
        value = self.prompt_critic_llm(idxs)
        self._update_actor_memory(reward, value, arguments, idxs)
        self._update_critic_memory(reward, value, idxs)
            
        
        # print(self.scratchpad.split('\n')[-1])
        self.step_n += 1
        print(self.step_n)
    
    
    
    
    def _build_agent_prompt(self, idxs) -> str:
        prompts = [self.agent_prompt.format(
                            examples = self.react_examples,
                            trajs = '', 
                            reflections = self.reflections_str[id],
                            question = self.task[id],
                            scratchpad = truncate_scratchpad(self.scratchpad[id],tokenizer=self.enc)) for id in idxs]

        return prompts 
    
    def _build_critic_prompt(self, idxs) -> str:
        history_list = []
        instruction_list = []
        for i, id in enumerate(idxs):
            temp_list = (self.task.get_history_actions(id)+self.argument_lists[id])[-10:]
            history_list.append(temp_list)
            instruction_list.append(self._get_critic_memory(temp_list)) 
            
        prompts = [self.critic_prompt.format(
                            history_list = history_list[i],
                            instruction = instruction_list[i]) for i, id in enumerate(idxs)]
        return prompts
    
    def _build_info(self, idxs) -> str:
        for id in idxs:
            userid = self.userids[id]
            self.infos[id] = {}
            prompt = self.agent_prompt.format(
                                examples = self.react_examples,
                                reflections = '',
                                trajs = '',
                                question = '',
                                scratchpad = '')
            traj = self.task[id] + self.scratchpad[id]
            # reflection = format_reflections(self.reflections[id], MAX=1000)
            self.infos[id].update({'userid': userid, 'prompt': prompt, 'traj': traj, 'traj_by_line': traj.split('\n')})
        
    
    def _update_memory(self, idxs, alpha = 0.5):
        embeddings = OpenAIEmbeddings()
        self.faiss_actor_memory = FAISS.from_texts(self.actor_memory.keys(), embeddings)
        self.faiss_critic_memory = FAISS.from_texts(self.critic_memory.keys(), embeddings)
    
    def _update_actor_memory(self, reward, value, arguments, idxs, gamma=0.5):
        for i, id in enumerate(idxs):
            try:
                v_i = float(value[i])
            except:
                v_i = 0
            temp_list = self.task.get_history_actions(id)+self.argument_lists[id][:-1]
            query = format_query(temp_list)
            if query not in self.actor_memory:
                self.actor_memory[query] = {}
            if reward + gamma*v_i - self.value_lists[id][-1] >= 0:
                self.actor_memory[query][arguments[i]] = 1
            else:
                self.actor_memory[query][arguments[i]] = -1
            self.value_lists[id].append(float(value[i]))
    
    def _update_critic_memory(self, reward, value, idxs, gamma=0.5):
        for i, id in enumerate(idxs):
            temp_list = self.task.get_history_actions(id)+self.argument_lists[id][:-1]
            query = format_query(temp_list)
            self.critic_memory[query] = reward + gamma * value[i]
    
    def _update_reflections_lib(self):
        embeddings = OpenAIEmbeddings()
        self.faiss_reflections = FAISS.from_texts(self.reflections, embeddings)
    
    def _get_actor_memory(self, history_list, argument_list, k=1):
        temp_list = history_list + argument_list
        query = format_query(temp_list)
       
        # keys = self.faiss_actor_memory.similarity_search_with_score(query, k=k)
        keys = try_with_delay(self.faiss_actor_memory, query, k)
        
        Q_values = [_ for key, score in keys for _ in self.actor_memory[key.page_content].values() if (score<0.5 and score>0.3) or score <0.01]
        actions = [_ for key, score in keys for _ in self.actor_memory[key.page_content].keys() if (score<0.5 and score>0.3) or score <0.01]
        
        pos_actions = [actions[i] for i in range(len(Q_values)) if Q_values[i]>0]
        neg_actions = [actions[i] for i in range(len(Q_values)) if Q_values[i]<0]
        if len(actions) == 0:
            return ''
        else:
            return f"According to historical experience, When {query}, we encourage to recommend {','.join(map(str, pos_actions))} items and not to recommend {','.join(map(str, neg_actions))} items."
    
    def _get_critic_memory(self, history_list, k=1):
        query = format_query(history_list)
        if self.faiss_critic_memory == None:
            return ''
        # keys = self.faiss_critic_memory.similarity_search_with_score(query, k=k)
        keys = try_with_delay(self.faiss_critic_memory, query, k)
        values = [self.critic_memory[key.page_content] for key, score in keys if score < 0.01]
        
        if len(values) == 0:
            return ''
        else:
            return f"According to historical experience, When {query}, the Value is {values[0]}"
        
        
    def format_reflections(self, reflections: List[str], id,
                        header: str = REFLECTION_HEADER, MAX=2) -> str:
        if reflections == []:
            return ''
        elif MAX ==0:
            return ''
        elif len(reflections) <= MAX:
            return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])
        else:
            # most similar strategy
            # docs = self.faiss_reflections.similarity_search(self.task[id], k=MAX)
            docs = try_with_delay(self.faiss_reflections, self.task[id], MAX)
            return header + 'Reflections:\n- ' + '\n- '.join([r.page_content.strip() for r,score in docs])
    
    def prompt_critic_llm(self, idxs):
        return format_step(self.critic_llm(self._build_critic_prompt(idxs)))
    
    
    

        
   
### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

def try_with_delay(memory, query, k):
    while True:
        try:
            result = memory.similarity_search_with_score(query, k=k)
            break
        except openai.error.AuthenticationError as e:
            print(f'c:{e}')
            time.sleep(10)
    return result


def format_step(steps: list) -> list:
    return [step.strip('\n').strip().replace('\n', '') for step in steps]

def format_query(argument_list, Max=10):
    last_elements = argument_list[-Max:]  # 获取最后十个元素
    result = ','.join(map(str, last_elements)) 
    query = 'The user viewing history is [' + result + ']'
    return query

def calculate_q_value(reward_list, gamma=0.5):
    q_value_list = [0] * len(reward_list)
    for i in range(len(reward_list)-1, -1, -1):
        if i == len(reward_list)-1:
            q_value_list[i] = reward_list[i]
        else:
            q_value_list[i] = reward_list[i] + gamma * q_value_list[i+1]
    return q_value_list
    
def format_last_attempt(question: str,
                        scratchpad: str,
                        header: str = LAST_TRIAL_HEADER):
    return header + f'Question: {question}\n' + truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'


def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)