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
from Agents.prompts import reflect_prompt, revise_prompt, react_agent_prompt, react_reflect_agent_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from Agents.prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt, COT_INSTRUCTION, COT_REFLECT_INSTRUCTION
from Agents.fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT
from Agents.agent_base import ReactAgent, parse_action, format_step, truncate_scratchpad
from Agents.agent_reflexion import ReactReflectAgent
from collections import defaultdict

class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context 
    REFLEXION: Apply reflexion to the next reasoning trace 
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace 
    """
    NONE = 'base'
    REFLEXION = 'reflexion'



class ReactReflectReviseAgent(ReactReflectAgent):
    def __init__(self,
                 task,
                 idxs: list, 
                 args, 
                 rec_env,
                 grounding_model,
                 max_steps: int = 30,
                 agent_prompt: PromptTemplate = react_reflect_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 revise_prompt: PromptTemplate = revise_prompt,
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
                 ) -> None:
        
        super().__init__(task, idxs, args, rec_env, grounding_model, max_steps=max_steps, agent_prompt = agent_prompt, react_llm=react_llm, reflect_llm=reflect_llm)
        
        self.revise_prompt = revise_prompt
    
    def step(self, idxs) -> None:
        # Think
        for id in idxs:
            self.scratchpad[id] += f'\nThought {self.step_n}:'
        thoughts = self.prompt_agent(idxs)
        for i, id in enumerate(idxs):
            self.scratchpad[id] += ' ' + thoughts[i]
        # print(self.scratchpad.split('\n')[-1])
        

        # Act
        for id in idxs:
            self.scratchpad[id] += f'\nAction {self.step_n}:'
        for _ in range(5):
            try:
                action = self.prompt_agent(idxs)
                action_types, arguments = parse_action(action)
                break
            except:
                continue
            
        self.scratchpad[id] += ' ' + action[i]
        for i, id in enumerate(idxs):
            if action_types[i] == 'recommend':
                old_film = arguments[i]
                
                argument_candidate = self.grounding_model.get_topk_near_item([old_film], self.args.Max_Iteration)[0]
                for argum in argument_candidate:
                    if argum not in self.argument_lists[id]:
                        arguments[i] = argum
                        break
                    
                if old_film != arguments[i]:
                    self.scratchpad[id] += f'\nAction {self.step_n}: [{old_film}] can not be recommened, instead, recommend[{arguments[i]}]'
                
        # print(self.scratchpad.split('\n')[-1])
        
        # Revise
        for id in idxs:
            self.scratchpad[id] += f'\nRevision {self.step_n}:'
        
        for _ in range(5):
            try:
                revision = self.revise_agent(idxs, arguments, thoughts)
                break
            except:
                continue
        for i, id in enumerate(idxs):
            print(revision[i])
            if 'Bad recommendation' in revision[i]:
                # bad recommendation
                print(revision[i])
                revision_action_types, revision_arguments = parse_action([revision[i]])
                print(revision_arguments)
                old_film = revision_arguments[0]
                print(old_film)
                argument_candidate = self.grounding_model.get_topk_near_item([old_film], self.args.Max_Iteration)[0]
                for argum in argument_candidate:
                    if argum not in self.argument_lists[id]:
                        arguments[i] = argum
                        break
                self.argument_lists[id].append(arguments[i])
                revision[i] = revision[i].replace(old_film, arguments[i])
                # print(f'old film {old_film} is replaced as {arguments[i]}.')
                self.scratchpad[id] += ' ' + revision[i]
            else:
                # good recommendation
                self.argument_lists[id].append(arguments[i])
                self.scratchpad[id] += ' ' + revision[i]
                
        # Observe
        for i, id in enumerate(idxs):
            self.scratchpad[id] +=  f'\nObservation {self.step_n}: '
        
            if action_types[i] == 'recommend':
                reward = self.env.get_reward(self.userids[id], arguments[i])
                if self.env.whether_to_leave(self.userids[id], arguments[i], self.argument_lists[id]):
                    self.scratchpad[id] += f"Episode finished, User Stop, reward={reward}"
                    self.finished[id] = True
                else:
                    self.scratchpad[id] += f"Episode continue, reward={reward}"

            else:
                self.scratchpad[id] += 'Invalid Action. Valid Actions are recommend[item].'
                self.finished[id] = True

        # print(self.scratchpad.split('\n')[-1])
        self.step_n += 1
        print(self.step_n)
    
    def revise_agent(self, idxs, action, thoughts) -> str:
        return format_step(self.llm(self._build_revise_prompt(idxs, action, thoughts)))
        
    def _build_revise_prompt(self, idxs, action, thoughts):
        promtps = [self.revise_prompt.format(
                            reflection=self.reflections_str[id], 
                            history=self.argument_lists[id], 
                            new_recommendation=action[i], 
                            thought=thoughts[i]) for i,id in enumerate(idxs)]
        return promtps
    
    



### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")


def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER, MAX=2) -> str:
    if reflections == [] or MAX == 0:
        return ''
    elif len(reflections) <= MAX:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])
    else:
        # simple strategy
        selected_reflections = reflections[-MAX:]
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in selected_reflections])

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