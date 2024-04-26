from langchain.prompts import PromptTemplate

COT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You will be given context that you should use to help you answer the question.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
Relevant Context: {context} 
Question: {question}{scratchpad}"""

COT_AGENT_REFLECT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You will be given context that you should use to help you answer the question.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Relevant Context: {context}
Question: {question}{scratchpad}"""

COT_REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to relevant context and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Relevant Context: {context}
Question: {question}{scratchpad}

Reflection:"""

cot_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "context", "question", "scratchpad"],
                        template = COT_INSTRUCTION,
                        )

cot_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "context", "question", "scratchpad"],
                        template = COT_AGENT_REFLECT_INSTRUCTION,
                        )

cot_reflect_prompt = PromptTemplate(
                        input_variables=["examples", "context", "question", "scratchpad"],
                        template = COT_REFLECT_INSTRUCTION,
                        )

COT_SIMPLE_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
{context}
Question: {question}{scratchpad}"""

COT_SIMPLE_AGENT_REFLECT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.
Here are some examples:
{examples}
(END OF EXAMPLES)
{context}
{reflections}

Question: {question}{scratchpad}"""

COT_SIMPLE_REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
Here are some examples:
{examples}
(END OF EXAMPLES)
{context}
Previous trial:
Question: {question}{scratchpad}

Reflection:"""

cot_simple_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "reflections", "context", "scratchpad"],
                        template = COT_SIMPLE_INSTRUCTION,
                        )

cot_simple_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "context", "reflections", "question", "scratchpad"],
                        template = COT_SIMPLE_AGENT_REFLECT_INSTRUCTION,
                        )

cot_simple_reflect_prompt = PromptTemplate(
                        input_variables=["examples", "question", "context", "scratchpad"],
                        template = COT_SIMPLE_REFLECT_INSTRUCTION,
                        )


REACT_INSTRUCTION = """Solve a recommendation task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation and current user interest, and 
Action can be the following types: 
(1) recommend[item], which recommend an item to user based on user's interest. Your goal is to meet the user's interest as much as possible and make recommendations to users as many times as possible. Note that if the user is not satisfied with your recommendations, he will quit and not accept new recommendations\n
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Task: {question}{scratchpad}"""

REACT_REFLECT_INSTRUCTION = """Solve a recommendation task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation and current user interest, and 
Action can be the following types: 
(1) recommend[item], which recommend an item to user based on user's interest. Your goal is to meet the user's interest as much as possible and make recommendations to users as many times as possible. Note that if the user is not satisfied with your recommendations, he will quit and not accept new recommendations\n
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""

REACT_REFLECT_RETRIVAL_INSTRUCTION = """Solve a recommendation task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation and current user interest, and 
Action can be the following types: 
(1) recommend[item], which recommend an item to user based on user's interest. Your goal is to meet the user's interest as much as possible and make recommendations to users as many times as possible. Note that if the user is not satisfied with your recommendations, he will quit and not accept new recommendations\n
You may take as many steps as necessary.
Here are some examples:
{examples}

{trajs}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""

REFLECTION_HEADER = 'You have attempted to recommend items to users before. The following reflection(s) give some insights on how to better interact with users. Use them to improve your strategy of recommending.\n'
REFLECTION_AFTER_LAST_TRIAL_HEADER = 'The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\n'

REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were required to recommend items for the user. You were unsuccessful in retaining multiple interaction because user does not satisfied with the recommendation. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure.  
Here are some examples:
{examples}

Previous trial:
Question: {question}{scratchpad}

Reflection:"""

REVISE_INSTRUCTION = """You are an advanced recommending system that follow the high level recommendation plan to recommend items to users. You need to determine whether this recommendation fits into the high-level plan.
Your answer should be the following types:
(1) Good recommendation, no revision is needed.
(2) Bad recommendation, recommend[item].

Here are an example:
Reflection from other's recommendation:
- The user has become tired of watching the same movie multiple times and is not satisfied with the repeated recommendation. The next time you recommend to a user, consider whether the user will be interested in watching the same movie again and focus on providing a variety of options that align with their preferences. Additionally, consider their previous viewing history and try to recommend movies that they have not watched yet to provide a fresh and diverse selection.
- The user did not respond positively to the second recommendation, indicating that they may not be interested in movies with a unique and artistic approach. The next time you recommend to a user, consider their previous feedback and adjust the recommendation accordingly. Additionally, it may be beneficial to provide a wider range of options that still align with their preferences for drama and psychological thrillers.

The user's view history: ['Mad Max 2', 'The Pawnbroker', 'The Paper Chase', 'Prince of the City', 'Serpico', 'The Godfather', 'Sunset Blvd.', 'The Third Man', 'Touch of Evil', 'The Gold Rush', 'The Maltese Falcon', 'Lawrence of Arabia', 'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb', 'The Manchurian Candidate', 'Grand Illusion', 'Chinatown', 'The Usual Suspects', 'Braveheart', 'Schindler's List', 'The Secret Agent', 'Microcosmos', 'Waterworld']
The high-level plan: The user is not satisfied with the recommendations in the Action and Adventure genre. I should try recommending a different genre. You should never recommend an item that the user has seen.
This round recommendation:[Batman Forever]
Revision:Bad recommendation, recommend[The Silence of the Lambs]

Reflection from other's recommendation: {reflection}

The user's view history: {history}
This round recommendation:[{new_recommendation}]
the high-level plan: {thought}. You should never recommend an item that the user has seen.
Revision:
"""

CRITIC_INSTRCUT = '''
You are an expert in recommendation systems. You need to determine whether the recommended list the recommendation list can satisfy the user.
Your answer must be a integer range from -5 to 5. (The higher the score, the higher the quality of the recommendation sequence, the more satisfying the user's interest, and the less annoying the user.)

Here are some examples:
Actor: The user's viewing history is ['Blockland', 'Shovel Knight: Treasure Trove', 'PlanetSide 2', 'Eldritch', \"Shantae: Risky's Revenge - Director's Cut\"]\n
Critic: 5\n

Actor: The user's viewing history is ['Blockland', 'Shovel Knight: Treasure Trove', 'PlanetSide 2', 'Eldritch', \"Shantae: Risky's Revenge - Director's Cut\", 'Plain Sight', 'The Path', 'Post Mortem']\n
Critic: 3\n

Actor: The user's viewing history is ['Blockland', 'Shovel Knight: Treasure Trove', 'PlanetSide 2', 'Eldritch', \"Shantae: Risky's Revenge - Director's Cut\", 'Plain Sight', 'The Path', 'Post Mortem', 'The Beginner's Guide']\n
Critic: -5\n
<End Example>

Actor: The user's viewing history is {history_list}\n
Instrction: {instruction}\n
Critic: 
'''

react_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = REACT_INSTRUCTION,
                        )

react_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "question", "scratchpad"],
                        template = REACT_REFLECT_INSTRUCTION,
                        )

react_reflect_retrival_agent_prompt = PromptTemplate(
                        input_variables=["examples", "trajs", "reflections", "question", "scratchpad"],
                        template = REACT_REFLECT_RETRIVAL_INSTRUCTION,
                        )

reflect_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = REFLECT_INSTRUCTION,
                        )

revise_prompt = PromptTemplate(
                        input_variables=["reflection", "history", "new_recommendation", "thought"],
                        template = REVISE_INSTRUCTION,
                        )

critic_prompt = PromptTemplate(
                        input_variables=["history_list", "instruction"],
                        template = CRITIC_INSTRCUT,
                        )
