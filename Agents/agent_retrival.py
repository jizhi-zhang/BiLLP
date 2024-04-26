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
from Agents.prompts import reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, react_reflect_retrival_agent_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from Agents.prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt, COT_INSTRUCTION, COT_REFLECT_INSTRUCTION
from Agents.fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT
from Agents.agent_base import ReactAgent, parse_action, format_step, truncate_scratchpad
from Agents.agent_reflexion import ReactReflectAgent, ReflexionStrategy




class ReactReflectRetrivalAgent(ReactReflectAgent):
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
                 reflections_memory = None,
                 Q_memory = None,
                 ) -> None:
        
        super().__init__(task, idxs, args, rec_env, grounding_model, max_steps, agent_prompt, react_llm, reflect_llm)
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS
        
        self.reflections_str: dict = {}
        
        
        if reflections_memory == None:
            self.reflections: list = []
            self.faiss_reflections = None
        else:
            self.reflections = reflections_memory
            self._update_reflections_lib()
        
        if Q_memory == None:
            self.Q_table: dict ={} 
            self.faiss_Q_table = None
        else:
            self.Q_table = Q_memory
            embeddings = OpenAIEmbeddings()
            self.faiss_Q_table = FAISS.from_texts(self.Q_table.keys(), embeddings)

        self.infos = {}
        self.final_infos = {}
        
        self.batch_size = args.batch_size
        self.enc = tiktoken.encoding_for_model("text-davinci-003")
    
    def run(self, reset = True, reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        
        for i in range(0, len(self.idxs), self.batch_size):
            temp_idxs = self.idxs[i: i+self.batch_size]
            print(f'temp_idxs:{temp_idxs}')
            
            self.get_reflect_str(reflect_strategy, temp_idxs)
            
            self.single_run(temp_idxs, reset)

            self.reflect(reflect_strategy, temp_idxs)
            self._update_reflections_lib()
            self._update_Q_lib(temp_idxs)
        
            self._build_info(temp_idxs)
        
        self.final_infos['trajs'] = self.infos
        self.final_infos['reflections'] = self.reflections
        self.final_infos['Q_table'] = self.Q_table
        return self.final_infos
    

    def _build_agent_prompt(self, idxs) -> str:
        prompts = [self.agent_prompt.format(
                            examples = self.react_examples,
                            trajs = '', 
                            reflections = self.reflections_str[id],
                            question = self.task[id],
                            scratchpad = truncate_scratchpad(self.scratchpad[id],tokenizer=self.enc)) for id in idxs]

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
        
    
    def _update_Q_lib(self, idxs, alpha = 0.5):
        # Q-learning table
        for id in idxs:
            length = len(self.argument_lists[id])
            q_value_list = calculate_q_value(self.reward_lists[id])
            for i in range(1, length):
                temp_list = self.argument_lists[id][:i]
                action = self.argument_lists[id][i]
                q_value = q_value_list[i]
                query = format_Q_record(temp_list)
                if query in self.Q_table:
                    if action in self.Q_table[query]:
                        self.Q_table[query][action] = alpha* self.Q_table[query][action] + (1-alpha)*q_value
                    else:
                        self.Q_table[query][action] = q_value
                else:
                    self.Q_table[query] = {action: q_value}
        
        
        embeddings = OpenAIEmbeddings()
        self.faiss_Q_table = FAISS.from_texts(self.Q_table.keys(), embeddings)
        
    
    def _update_reflections_lib(self):
        embeddings = OpenAIEmbeddings()
        self.faiss_reflections = FAISS.from_texts(self.reflections, embeddings)
    
    def _get_Q_record(self, history_list, argument_list, k=5):
        temp_list = history_list + argument_list
        query = format_Q_record(temp_list)
        keys = self.faiss_Q_table.similarity_search(query, k=k)
        Q_values = [_ for key in keys for _ in self.Q_table[key.page_content].values()]
        actions = [_ for key in keys for _ in self.Q_table[key.page_content].keys()]
        
        max_index = max(range(len(Q_values)), key=Q_values.__getitem__) 
        min_index = min(range(len(Q_values)), key=Q_values.__getitem__) 
        
        return f"According to Q-learning table, we encourage to recommend {actions[max_index]} items and not to recommend {actions[min_index]} items."
            
    def format_reflections(self, reflections: List[str], id,
                        header: str = REFLECTION_HEADER, MAX=2) -> str:
        if reflections == []:
            return ''
        elif MAX ==0:
            return ''
        elif len(reflections) <= MAX:
            return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])
        else:
            # simple strategy
            # selected_reflections = reflections[-MAX:]
            # return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in selected_reflections])
            
            # most similar strategy
            docs = self.faiss_reflections.similarity_search(self.task[id], k=MAX)
            return header + 'Reflections:\n- ' + '\n- '.join([r.page_content.strip() for r in docs])
            
   
### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

def format_Q_record(argument_list, Max=10):
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