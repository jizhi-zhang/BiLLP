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
from langchain.prompts import PromptTemplate
from Agents.llm import AnyOpenAILLM
from Agents.prompts import reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from Agents.prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt, COT_INSTRUCTION, COT_REFLECT_INSTRUCTION
from Agents.fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT
from Agents.agent_base import ReactAgent, parse_action, format_step, truncate_scratchpad
from collections import defaultdict
import random

class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context 
    REFLEXION: Apply reflexion to the next reasoning trace 
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace 
    """
    NONE = 'base'
    REFLEXION = 'reflexion'



class ReactReflectAgent(ReactAgent):
    def __init__(self,
                 task,
                 idxs: list, 
                 args, 
                 rec_env,
                 grounding_model,
                 max_steps: int = 30,
                 agent_prompt: PromptTemplate = react_reflect_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 react_llm: AnyOpenAILLM = AnyOpenAILLM(
                                             temperature=0,
                                             max_tokens=3000,
                                             model_name="gpt-3.5-turbo-16k",
                                             model_kwargs={"stop": "\n"},
                                             openai_api_key=os.environ['OPENAI_API_KEY'],
                                             openai_api_base = os.environ['OPENAI_API_BASE']),
                 reflect_llm: AnyOpenAILLM = AnyOpenAILLM(
                                               temperature=0,
                                               max_tokens=3000,
                                               model_name="gpt-3.5-turbo-16k",
                                               openai_api_key=os.environ['OPENAI_API_KEY'],
                                               openai_api_base = os.environ['OPENAI_API_BASE']),
                 reflections_memory = None,
                 ) -> None:
        
        super().__init__(task, idxs, args, rec_env, grounding_model, max_steps, agent_prompt, react_llm)
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS
        if reflections_memory == None:
            self.reflections: list = []
        else:
            self.reflections = reflections_memory
        self.reflections_str: dict = {}
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
            
            self._build_info(temp_idxs)
        
        self.final_infos['trajs'] = self.infos
        self.final_infos['reflections'] = self.reflections
        return self.final_infos
    
    
    def get_reflect_str(self,
                strategy: ReflexionStrategy, idxs) -> None:
        print('Reflecting...')
        
        if strategy == ReflexionStrategy.REFLEXION: 
            for i, id in enumerate(idxs):
                self.reflections_str[id] = self.format_reflections(self.reflections, id, MAX=self.args.Max_Reflections)  
                print(self.reflections_str[id])
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        

        
    def reflect(self,
                strategy: ReflexionStrategy, idxs) -> None:
        print('Reflecting...')
        
        if strategy == ReflexionStrategy.REFLEXION: 
            self.reflections += self.prompt_reflection(idxs)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
    
    def prompt_reflection(self, idxs) -> str:
        return format_step(self.reflect_llm(self._build_reflection_prompt(idxs)))


    def _build_reflection_prompt(self, idxs) -> str:
        prompts = [self.reflect_prompt.format(
                            examples = self.reflect_examples,
                            question = self.task[id],
                            scratchpad = self.scratchpad[id]) for id in idxs]
        return prompts
 
    def _build_agent_prompt(self, idxs) -> str:
        prompts = [self.agent_prompt.format(
                            examples = self.react_examples,
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
                                question = '',
                                scratchpad = '')
            traj = self.task[id] + self.scratchpad[id]
            # reflection = format_reflections(self.reflections[id], MAX=1000)
            self.infos[id].update({'userid': userid, 'prompt': prompt, 'traj': traj, 'traj_by_line': traj.split('\n')})
    
    def format_reflections(self, reflections: List[str], id,
                        header: str = REFLECTION_HEADER, MAX=2) -> str:
        if reflections == []:
            return ''
        elif MAX == 0:
            return ''
        elif len(reflections) <= MAX:
            return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])
        else:
            # simple strategy randomly chosen
            random_index = random.sample(list(range(len(reflections))), MAX)
            selected_reflections = [reflections[i] for i in random_index]
            return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in selected_reflections])
        


### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")




def format_last_attempt(task: str,
                        idxs, 
                        scratchpad: str,
                        header: str = LAST_TRIAL_HEADER):
    strs = {}
    for id in idxs:
        strs[id] =  header + f'Question: {task[id]}\n' + truncate_scratchpad(scratchpad[id], tokenizer=gpt2_enc).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'
    return strs


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