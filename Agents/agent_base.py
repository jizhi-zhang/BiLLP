import re, string, os
from typing import List, Union, Literal
from enum import Enum
import random
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
from Agents.fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT, WEBTHINK_SIMPLE7, WEBTHINK_SIMPLE8, WEBTHINK_SIMPLE9, WEBTHINK_SIMPLE10, WEBTHINK_SIMPLE11
from collections import defaultdict

GENRE_MOVIE = ['Animation', "Children's", 'Comedy', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Sci-Fi', 'Documentary', 'War', 'Musical', 'Mystery', 'Film-Noir', 'Western']
GENRE_STEAM = ['Accounting', 'Simulation', 'Adventure', 'Free to Play', 'Massively Multiplayer', 'Sports', 'Racing', 'Indie', 'Software Training', 'Early Access', 'Strategy', 'RPG', 'Action', 'Education', 'Casual']
GENRE_AMAZON = ['Self-Esteem', 'Gaming', 'British & Irish', 'Business Technology', 'Graphic Design', 'Television', 'Words, Language & Grammar', 'Antiques & Collectibles', 'Business & Money', 'Evolution', 'Mountaineering', 'Education & Reference', 'Outdoor Cooking', 'Personal Transformation', 'History', 'Historical Study & Educational Resources', 'Middle East', 'Memoirs', 'Northeast', 'Science Fiction & Fantasy', 'Bible Study & Reference', 'Worship & Devotion', 'Cooking by Ingredient', 'Comic Books', 'Fairy Tales, Folk Tales & Myths', 'Pets & Animal Care', 'Australia & Oceania', 'Publishers', 'Social Sciences', 'Kitchen Appliances', 'Law Practice', 'Special Diet', 'Success', 'Fashion', 'Travel', 'Astronomy & Space Science', "Children's Books", 'Science, Nature & How It Works', 'Mysteries & Thrillers', 'Business', 'Reference', 'Cookbooks, Food & Wine', 'Sports & Outdoors', 'Action & Adventure', 'History & Philosophy', 'Travel Writing', 'Motivational', 'Death & Grief', 'Administration & Medicine Economics', 'Other Media', 'Photography & Video', 'Buddhism', 'Movies', 'Religious Studies', 'Nature & Ecology', 'Main Courses & Side Dishes', 'Beverages & Wine', 'Bibles', 'Ancient & Medieval Literature', 'Puzzles & Games', 'Constitutional Law', 'Canning & Preserving', 'Australia & South Pacific', 'Christian Books & Bibles', 'Needlecrafts & Textile Crafts', 'Mystery & Thrillers', 'Miscellaneous', 'Time Travel', 'New Age & Spirituality', 'Family Relationships', 'Hinduism', "Children's Health", 'Biography & History', 'Software', 'Business Culture', 'Diets & Weight Loss', 'Lesbian, Gay, Bisexual & Transgender Books', 'Americas', 'Hunting & Fishing', 'Mystery', 'Military', 'Biographies', 'Addiction & Recovery', 'Literary', 'Creativity', 'Paranormal & Urban', 'Graphic Novels', 'Industries', 'Stress Management', 'Historical Fiction', 'Historical', 'Atheism', 'Performing Arts', 'Parenting', 'Health, Fitness & Dieting', 'Automotive', 'Dictionaries & Thesauruses', 'Hiking & Camping', 'Processes & Infrastructure', 'Business & Finance', 'Writing, Research & Publishing Guides', 'Test Preparation', 'Clean & Wholesome', 'Hypnosis', 'Gardening & Landscape Design', 'Holidays', 'Leaders & Notable People', 'Thrillers & Suspense', 'Catholicism', 'Happiness', 'New England', 'Computers & Technology', 'Short Stories & Anthologies', 'Computer Science', 'Criminal Law', 'Arts, Music & Photography', 'Behavioral Sciences', 'Other Diets', 'Romantic Comedy', 'Anthologies', 'Home Improvement & Design', 'Teen & Young Adult', 'Cooking Education & Reference', 'Medical Books', 'Atlases & Maps', 'Job Hunting & Careers', 'Individual Artists', 'Architecture', 'Judaism', 'United States', 'Schools & Teaching', 'Regency', 'Ethnic & National', 'Dramas & Plays', 'Diseases & Physical Ailments', 'Programming', 'Operating Systems', 'Biological Sciences', 'Psychology', 'Engineering & Transportation', 'Engineering', 'Russia', 'Hardware & DIY', 'Science & Math', 'Fantasy', 'Africa', 'Marketing & Sales', 'Songbooks', 'Medicine & Health Sciences', 'Individual Sports', 'Small Business & Entrepreneurship', 'Drawing', 'Catalogs & Directories', 'Real Estate', 'Biographies & Memoirs', 'Poetry', 'Golf', 'Guitars & Fretted Instruments', 'Management & Leadership', 'Humor & Satire', 'Water Sports', 'Transportation', 'Humorous', 'Law', 'New, Used & Rental Textbooks', 'Comic Strips', 'Regional & International', 'LGBT', 'Religion & Spirituality', 'Activities, Crafts & Games', 'Sociology', 'Experiments, Instruments & Measurement', 'Nutrition', 'Christian Denominations & Sects', 'Churches & Church Leadership', 'Alternative Medicine', 'Canada', 'Games & Strategy Guides', 'South America', 'Animals', 'Ministry & Evangelism', 'Baking', 'War', 'Decorative Arts & Design', 'Protestantism', 'Early Learning', 'Coaching', 'Studying & Workbooks', 'Music', 'Etiquette', 'Politics & Social Sciences', 'Science & Mathematics', 'Europe', 'Humanities', 'Western', 'College & High School', 'Networking & Cloud Computing', 'Arts & Photography', 'Geography & Cultures', 'Science Fiction', 'Investing', 'Earth Sciences', 'Baseball', 'Asia', 'Occult & Paranormal', 'Physics', 'Regional U.S.', '</span>', 'Economics', 'Anthropology', 'Contemporary', 'Arts & Literature', 'Foreign Language Study & Reference', 'Vampires', 'True Crime', 'Romance', 'Specific Groups', 'Cars, Trains & Things That Go', 'Crafts, Hobbies & Home', 'Psychology & Counseling', 'Mathematics', 'Other Religions, Practices & Sacred Texts', 'Finance', 'Professionals & Academics', 'Sports', 'Religions', 'Archaeology', 'Chemistry', 'Web Development & Design', 'Crafts & Hobbies', 'Coming of Age', 'Holidays & Celebrations', 'Food, Lodging & Transportation', 'Classics', 'Administrative Law', 'Cooking Methods', 'Education & Teaching', 'How To Create Comics & Manga', 'History & Criticism', 'Databases & Big Data', 'Genre Fiction', 'Radio', 'Exercise & Fitness', 'Business & Professional Growth', 'Self-Help', 'Mental Health', "Women's Fiction", 'Humor & Entertainment', 'Comics & Graphic Novels', 'Mythology & Folk Tales', 'Ancient Civilizations', 'Ancient & Classical', 'Personal Health', 'Skills', 'Conflict Management', 'Theology', 'Field Guides', 'Safety & First Aid', 'Manga', 'Education', 'Mysteries & Detectives', 'Encyclopedias & Subject Guides', 'State & Local', 'Humor', 'Growing Up & Facts of Life', 'Parenting & Relationships', 'Christian Living', 'World Literature', 'New Adult & College', 'Gothic', 'Multicultural', 'Relationships', 'Erotica', 'Guitars', 'Romantic Suspense', 'Philosophy', 'Paranormal', 'Travelers & Explorers', 'Mystery, Thriller & Suspense', 'World', 'Other Eastern Religions & Sacred Texts', 'Medicine', 'Suspense', 'Urban Life', 'Politics & Government', 'Business & Professional', 'Personal Finance', 'Literature & Fiction', 'Sustainable Living', 'Essays & Correspondence']



class ReactAgent:
    def __init__(self,
                 task,
                 idxs: int, 
                 args, 
                 rec_env,
                 grounding_model,
                 max_steps: int = 30,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 react_llm: AnyOpenAILLM = AnyOpenAILLM(
                                            temperature=0,
                                            max_tokens=100,
                                            model_name="gpt-3.5-turbo-16k",
                                            model_kwargs={"stop": "\n"},
                                            openai_api_key=os.environ['OPENAI_API_KEY'],
                                            openai_api_base = os.environ['OPENAI_API_BASE']),
                 ) -> None:
        
        self.task = task
        self.idxs = idxs
        self.userids = {id: self.task.get_userid(id) for id in self.idxs}
        self.grounding_model = grounding_model
        self.answer = ''
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        self.react_examples = WEBTHINK_SIMPLE6
        if args.env == 'movie':
            self.react_examples = WEBTHINK_SIMPLE8
            self.GENRE = GENRE_MOVIE
        elif args.env == 'steam':
            self.react_examples = WEBTHINK_SIMPLE8
            self.GENRE = GENRE_STEAM
        elif args.env == 'amazon':
            self.react_examples = WEBTHINK_SIMPLE7
            self.GENRE = GENRE_AMAZON

    
        self.env = rec_env
        self.llm = react_llm
        self.args = args
        #self.enc = tiktoken.encoding_for_model("text-davinci-003")

        self.scratchpad: dict = defaultdict(str)
        # self.__reset_agent()
        
        

    def single_run(self, idxs, reset = True) -> None:
        if reset:
            self.__reset_agent(idxs)
        
        step_idxs = idxs
        while not self.is_halted() and not self.is_finished():
            self.step(step_idxs)
            step_idxs = self._filter_idx()
            
    def step(self, idxs) -> None:
        # Think
        for id in idxs:
            self.scratchpad[id] += f'\nThought {self.step_n}:'
        prompts = self.prompt_agent(idxs)
        for i, id in enumerate(idxs):
            self.scratchpad[id] += ' ' + prompts[i]
            
            
        # print(self.scratchpad.split('\n')[-1])
        random_type = []
        q_prompt = {}
        for i, id in enumerate(idxs):
            hist = self.env.get_hist_list(self.argument_lists[id])
            random_type.append(random.sample([x for x in self.GENRE if x not in hist], 2))
            # self.scratchpad[id] += f'(Do not recommend a certain type of movie over twice recently, such as these {hist} type. Please recommend {random_type[i][0]} movies to help users explore their interests)'
            # self.scratchpad[id] += f'Ignore the user interests, Please recommend {random_type[i][0]} and {random_type[i][1]} games to help users explore their interests'
            # self.scratchpad[id] += f'(Please recommend {random_type[i][0]} books to help users explore their interests)'
            self.scratchpad[id] += f'(Please recommend {random_type[i][0]} and {random_type[i][1]} items to help users explore their interests)'
        
            if self.args.agent_name == 'agent_retrival' and self.faiss_Q_table!=None:
                q_prompt[id] = self._get_Q_record(self.task.get_history_actions(id), self.argument_lists[id])
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
            if self.args.agent_name == 'agent_retrival' and self.faiss_Q_table!=None:
                self.scratchpad[id] = self.scratchpad[id].replace(q_prompt[id], '')
                
        
        for i, id in enumerate(idxs):
            
            self.scratchpad[id] += ' ' + action[i]
            
            if action_types[i] == 'recommend':
                old_film = arguments[i]
                argument_candidate = self.grounding_model.get_topk_near_item([old_film], self.args.Max_Iteration)[0]
                # for argum in argument_candidate:
                #     if argum not in self.argument_lists[id]:
                #         arguments[i] = argum
                #         break
                arguments[i] = argument_candidate[0]
                self.argument_lists[id].append(arguments[i])
                if old_film != arguments[i]:
                    self.scratchpad[id] += f'\nObservation {self.step_n}: [{old_film}] can not be recommened, instead, recommend[{arguments[i]}]'
                
                
        # print(self.scratchpad.split('\n')[-1])

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

        # print(self.scratchpad.split('\n')[-1])
        self.step_n += 1
        print(self.step_n)

    def prompt_agent(self, idxs) -> str:
        return format_step(self.llm(self._build_agent_prompt(idxs)))
    
    def _build_agent_prompt(self, idxs) -> str:
        promtps = [self.agent_prompt.format(
                            examples = self.react_examples,
                            question = self.task[id],
                            scratchpad = self.scratchpad[id]) for id in idxs]
        return promtps
    
    def is_finished(self) -> bool:
        for key, value in self.finished.items():
            if value == False:
                return False
        return True

    def is_halted(self) -> bool:
        return (self.step_n > self.max_steps)

    def __reset_agent(self, idxs) -> None:
        self.step_n = 1
        self.finished = {id: False if id in idxs else True for id in self.idxs}
        # self.argument_lists = {id: self.task.get_history_actions(id) for id in self.idxs}
        self.argument_lists = {id: [] for id in self.idxs}
        self.reward_lists = {id: [] for id in self.idxs}
        self.value_lists = {id: [0] for id in self.idxs}
    
    def _filter_idx(self):
        return [key for key, value in self.finished.items() if value == False]


def format_step(steps: list) -> list:
    return [step.strip('\n').strip().replace('\n', '') for step in steps]

### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

def parse_action(list_str: list[str]):
    pattern = r'^(\w+)\[(.+)\]$'
    
    action_types = []
    arguments = []
    
    for string in list_str:
        match = re.match(pattern, string)
        if match:
            action_types.append(match.group(1))
            arguments.append(match.group(2))
             
        else:
            action_types.append(None)
            arguments.append(None)
    return action_types, arguments
    
def truncate_scratchpad(scratchpad: str, n_tokens: int = 4000, tokenizer = gpt2_enc) -> str:
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Thought'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated thought]'
    return '\n'.join(lines)
