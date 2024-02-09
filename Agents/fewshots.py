WEBTHINK_SIMPLE6 = """
Question: The user's viewing history is ['Pretty in Pink', \"One Flew Over the Cuckoo's Nest\", 'Ransom', 'Saving Private Ryan', 'X-Men', 'Coyote Ugly', 'The Patriot', 'Me, Myself and Irene', 'Gone in 60 Seconds', 'The Perfect Storm', 'Titanic', 'The Haunting', 'Bedknobs and Broomsticks', 'Clerks', 'The Matrix', 'The Shawshank Redemption', 'Vacation', 'Father of the Bride', 'Wallace & Gromit: The Best of Aardman Animation', 'Back to the Future', 'Fight Club'], please recommend item for this user  \n
Thought: The user seems to enjoy a mix of drama, action, and comedy. They also seem to appreciate classic films. I should recommend a movie that fits these categories.  \n  
Action: recommend[Forrest Gump]  \n  
Observation: Episode continue, reward=0.30697035862193367  \n
Thought:The user seems to have responded positively to the previous recommendation. They seem to enjoy movies with a strong narrative and compelling characters. I should recommend another movie in a similar vein.  \n
Action: recommend[The Godfather]  \n
Observation: Episode finished, reward=0.49717313011547304  \n
"""

WEBTHINK_SIMPLE7 = """
Question: The user's viewing history is ['Pretty in Pink', \"One Flew Over the Cuckoo's Nest\", 'Ransom', 'Saving Private Ryan', 'X-Men', 'Coyote Ugly', 'The Patriot', 'Me, Myself and Irene', 'Gone in 60 Seconds', 'The Perfect Storm', 'Titanic', 'The Haunting', 'Bedknobs and Broomsticks', 'Clerks', 'The Matrix', 'The Shawshank Redemption', 'Vacation', 'Father of the Bride', 'Wallace & Gromit: The Best of Aardman Animation', 'Back to the Future', 'Fight Club'], please recommend item for this user\n
Thought: The user seems to enjoy a mix of drama, action, and comedy. They also seem to appreciate thriller and war films. I would first recommend the user his favorite drama genre movies, and then recommend some other niche genre movies that he likes.\n
Action: recommend[The Godfather]\n
Observation: Episode continue, reward=0.90697035862193367\n
Thought:The user seems to have responded positively to this drama and action type film. Based on previous plan, He likes comedy movies, too. I should give him another movie combining comedy and drama.\n
Action: recommend[The American President]\n
Observation: Episode continue, reward=0.81253526485664527\n
Thought: It seems that the users are satisfied with the recommendation. Following previous plan, after recommending his favorite genre of movies, we should recommend some other niche genre movies that he likes, such as thriller and war films.\n
Action: recommend[The Transformers: The Movie]\n
Observation: Episode continue, reward=0.22653369526521027\n
Thought: This recommendation seems not satisfy the user. This may because the movie was aimed at children, and the user didn't like it. In order to retain the user in the long term, we need to recommend his favorite drama movies.\n
Action: recommend[Braveheart]\n
Observation: Episode continue, reward=0.96525663552102725\n
Thought: The user like this recommendation. In next step, the new plan should involve considering the user's specific preferences and recommending a diverse range of movies across different genres.\n
Action: recommend[First Knight]\n
Observation: Episode finished, User Stop, reward=-1000.00000\n
"""

WEBTHINK_SIMPLE8 = """
Question: The user's viewing history is ['Pretty in Pink', \"One Flew Over the Cuckoo's Nest\", 'Ransom', 'Saving Private Ryan', 'X-Men', 'Coyote Ugly', 'The Patriot', 'Me, Myself and Irene', 'Gone in 60 Seconds', 'The Perfect Storm', 'Titanic', 'The Haunting', 'Bedknobs and Broomsticks', 'Clerks', 'The Matrix', 'The Shawshank Redemption', 'Vacation', 'Father of the Bride', 'Wallace & Gromit: The Best of Aardman Animation', 'Back to the Future', 'Fight Club'], please recommend item for this user\n
Thought: The user seems to enjoy a mix of drama, action, and comedy. They also seem to appreciate thriller and war films. I would first recommend the user his favorite drama genre movies, and then recommend some other niche genre movies that he likes.\n
Action: recommend[The Godfather]\n
Observation: Episode continue, reward=0.90697035862193367\n
Thought:The user seems to have responded positively to this drama and action type film. Based on previous plan, He likes comedy movies, too. I should give him another movie combining comedy and drama.\n
Action: recommend[The American President]\n
Observation: Episode continue, reward=0.81253526485664527\n
Thought: It seems that the users are satisfied with the recommendation. Following previous plan, after recommending his favorite genre of movies, we should recommend some other niche genre movies that he likes, such as thriller and war films.\n
Action: recommend[The Transformers: The Movie]\n
Observation: Episode continue, reward=0.22653369526521027\n
Thought: This recommendation seems not satisfy the user. This may because the movie was aimed at children, and the user didn't like it. In order to retain the user in the long term, we need to recommend his favorite drama movies.\n
Action: recommend[Braveheart]\n
Observation: Episode continue, reward=0.96525663552102725\n
Thought: The user like this recommendation. In next step, the new plan should involve considering the user's specific preferences and recommending a diverse range of movies across different genres.\n
Action: recommend[First Knight]\n
Observation: Episode finished, User Stop, reward=-1000.00000\n


Question: The user's viewing history is ['Blockland', 'Shovel Knight: Treasure Trove', 'PlanetSide 2', 'Eldritch', \"Shantae: Risky's Revenge - Director's Cut\"], please recommend item for this user
Thought: The user seems to enjoy a mix of Action and Independent video game. They also seem to appreciate Adventure games. I would first recommend the user his favorite action games, and then recommend some other niche genre games that he likes.\n
Action: recommend[Plain Sight]\n
Observation: Episode continue, reward=4\n
Thought: The user seems to have responded positively to this Action and Indid type games. Based on previous plan, He likes Adventure games, too. I should recommend him another Adventure game.\n
Action: recommend[The Path]\n
Observation: Episode continue, reward=5\n
Thought: It seems that the users are satisfied with the recommendation. Following previous plan, after recommending his favorite genre of games, we should recommend some other niche genre games that he likes, such as RPG games.\n
Action: recommend[I, Gladiator]\n
Observation: Episode continue, reward=4\n
Thought: The user like this recommendation. In next step, the new plan should involve considering the user's specific preferences and recommending a diverse range of games across different genres.\n
Action: recommend[Thirty Flights of Loving]\n
Observation: Episode finished, User Stop, reward=-1000.00000\n
"""


WEBTHINK_SIMPLE9 = """
Question: The user's viewing history is ['Blockland', 'Shovel Knight: Treasure Trove', 'PlanetSide 2', 'Eldritch', \"Shantae: Risky's Revenge - Director's Cut\"], please recommend item for this user
Thought: The user seems to enjoy a mix of Action and Independent video game. They also seem to appreciate Adventure games. I would first recommend the user his favorite action games, and then recommend some other niche genre games that he likes.\n
Action: recommend[Plain Sight]\n
Observation: Episode continue, reward=4\n
Thought: The user seems to have responded positively to this Action and Indid type games. Based on previous plan, He likes Adventure games, too. I should recommend him another Adventure game.\n
Action: recommend[The Path]\n
Observation: Episode continue, reward=5\n
Thought: It seems that the users are satisfied with the recommendation. Following previous plan, after recommending his favorite genre of games, we should recommend some other niche genre games that he likes, such as RPG games.\n
Action: recommend[I, Gladiator]\n
Observation: Episode continue, reward=4\n
Thought: The user like this recommendation. In next step, the new plan should involve considering the user's specific preferences and recommending a diverse range of games across different genres.\n
Action: recommend[Thirty Flights of Loving]\n
Observation: Episode finished, User Stop, reward=-1000.00000\n
"""


WEBTHINK_SIMPLE10 = """
The user's viewing history is ['Then Came You', 'The Prey', 'Forbidden', 'Whispers in the Dark', 'Dirty Laundry', 'Rebel', 'Muscle', 'Bittersweet', 'Housekeeping', 'Accidentally Married', 'Feral', 'Resonance', 'Sin', 'Protector', 'Three'], please recommend item for this user.\n
Thought 1:Please recommend Arts & Photography and Romance books to help users explore their interests\n
Action 1: recommend[The Art of Fielding]\n
Observation 1: Episode continue, reward=5\n
Thought 2:Please recommend Photography & Video and Science, Nature & How It Works books to help users explore their interests\n
Action 2: recommend[National Geographic: The Photo Ark]\n
Observation 2: [National Geographic: The Photo Ark] can not be recommened, instead, recommend[It's Just A Dog]\n
Observation 2: Episode continue, reward=4\n
Thought 3:Please recommend Self-Esteem and Personal Health books to help users explore their interests\n
Action 3: recommend[The Gifts of Imperfection]\n
Observation 3: Episode finished, User Stop, reward=-1000.000\n
"""

WEBTHINK_SIMPLE11 = """
Question: The user's viewing history is ['Pretty in Pink', \"One Flew Over the Cuckoo's Nest\", 'Ransom', 'Saving Private Ryan', 'X-Men', 'Coyote Ugly', 'The Patriot', 'Me, Myself and Irene', 'Gone in 60 Seconds', 'The Perfect Storm', 'Titanic', 'The Haunting', 'Bedknobs and Broomsticks', 'Clerks', 'The Matrix', 'The Shawshank Redemption', 'Vacation', 'Father of the Bride', 'Wallace & Gromit: The Best of Aardman Animation', 'Back to the Future', 'Fight Club'], please recommend item for this user\n
Thought: The user seems to enjoy a mix of drama, action, and comedy. They also seem to appreciate thriller and war films. I would first recommend the user his favorite drama genre movies, and then recommend some other niche genre movies that he likes.\n
Action: recommend[The Godfather]\n
Observation: Episode continue, reward=0.90697035862193367\n
Thought:The user seems to have responded positively to this drama and action type film. Based on previous plan, He likes comedy movies, too. I should give him another movie combining comedy and drama.\n
Action: recommend[The American President]\n
Observation: Episode continue, reward=0.81253526485664527\n
Thought: It seems that the users are satisfied with the recommendation. Following previous plan, after recommending his favorite genre of movies, we should recommend some other niche genre movies that he likes, such as thriller and war films.\n
Action: recommend[The Transformers: The Movie]\n
Observation: Episode continue, reward=0.22653369526521027\n
Thought: This recommendation seems not satisfy the user. This may because the movie was aimed at children, and the user didn't like it. In order to retain the user in the long term, we need to recommend his favorite drama movies.\n
Action: recommend[Braveheart]\n
Observation: Episode continue, reward=0.96525663552102725\n
Thought: The user like this recommendation. In next step, the new plan should involve considering the user's specific preferences and recommending a diverse range of movies across different genres.\n
Action: recommend[First Knight]\n
Observation: Episode finished, User Stop, reward=-1000.00000\n

Question: The user's viewing history is ['Reckless', 'Unseen', 'Born in Fire', 'Defender', 'In the Blood', 'An Eye for an Eye', 'Embrace', 'Dead in the Water', 'Hell or High Water', \"Another Man's Treasure\", 'Dominion', 'Double Dog Dare', 'Save Me', '15 Minutes', 'Forgotten'], please recommend item for this user.\n
Thought: The user seems to enjoy a mix of mystery, suspense, and perhaps action. I would first recommend the user his favorite mystery novels, and then recommend some other niche genre books that he likes.\n
Action: recommend[The Shadow of the Wind]\n
Observation: Episode continue, reward=0.98197085292213302\n
Thought: The user seems to have responded positively to this mystery type novels. Based on previous plan, after recommending his favorite genre, I should recommend some other niche genre books that he likes, such as suspense novels.\n
Action: recommend[The Silent Patient]\n\nObservation: Episode continue, reward=0.80817019162112308\n
Thought: The user likes this recommendation. Following previous plan, the user perhaps also enjoys action novels. I should recommend him an action novel now.\n
Action: recommend[The Reversal]\n\nObservation: Episode continue, reward=0.29612935893019367\n
Thought: The user doesn't like this recommendation. In order to retain the user in the long term, we need to recommend his favorite mystery novels.\n
Action: recommend[The Sweetness at the Bottom of the Pie]\n
Observation: Episode continue, reward=0.9418192090819617\n
Thought: The user likes this recommendation. In next step, the new plan should involve considering the user's specific preferences and recommending a diverse range of novels across different genres.\n
Action: recommend[The Hare with Amber Eyes]\n
Observation: Episode finished, User Stop, reward=-1000.00000\n
"""


REFLECTIONS = """
Previous Trial:
Question: The user's viewing history is ['The Patriot', 'Me, Myself and Irene', 'Gone in 60 Seconds', 'The Perfect Storm', 'Titanic', 'The Haunting', 'Bedknobs and Broomsticks', 'Clerks', 'The Matrix', 'The Shawshank Redemption', 'Vacation', 'Father of the Bride', 'Wallace & Gromit: The Best of Aardman Animation', 'Back to the Future', 'Fight Club'], please recommend item for this user
Thought 1: The user seems to enjoy a mix of genres, including action, comedy, and drama. They also seem to appreciate classic films. I should recommend a movie that fits these categories.
Action 1: recommend[Forrest Gump]
Observation 1: Episode continue, reward=0.9370001554489136
Thought 2: The user responded positively to the previous recommendation. They seem to enjoy movies with a strong narrative and compelling characters. I should recommend another movie in a similar vein.
Action 2: recommend[The Godfather]
Observation 2: Episode continue, reward=0.9768682718276978
Thought 3: The user seems to have a preference for classic films and movies with a strong narrative. I should recommend another movie that fits these criteria.
Action 3: recommend[The Shawshank Redemption]
Observation 3: Episode continue, reward=0.9870089888572693
Thought 4: The user has consistently responded positively to the recommendations so far. I should continue recommending movies that fit their preferences.
Action 4: recommend[The Dark Knight]
Observation 4: Episode finished, User Stop, reward=0.6157855987548828

Reflection: The user has become tired of this type of movie and is not satisfied with the final recommendation, so quit. The next time you recommend to a user, consider whether the user will get bored
"""

COTQA_SIMPLE6 = """
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: Let's think step by step. The eastern sector of Colorado orogeny extends into the High Plains. High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action: Finish[1,800 to 7,000 ft]

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: Let's think step by step. Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action: Finish[Richard Nixon]

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought: Let's think step by step. Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action: Finish[The Saimaa Gesture]

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: Let's think step by step. Professions of Nicholas Ray are director, screenwriter, and actor. Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action: Finish[director, screenwriter, actor]

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought: Let's think step by step. Arthur's Magazine was started in 1844. First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action: Finish[Arthur's Magazine]

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: Let's think step by step. Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.
Action: Finish[Yes]"""

COT_SIMPLE_REFLECTION = """
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: Let's think step by step. The eastern sector of Colorado orogeny extends into the Rocky Mountains. The Rocky Mountains rise in elevation from around 1,800 to 14,000 ft, so the answer is 1,800 to 14,000 ft.
Action: Finish[1,800 to 14,000 ft]

Reflection: My reasoning for the elevation range of the eastern sector of the Colorado orogeny failed because I incorrectly associated it with the Rocky Mountains instead of the High Plains. In the future, when attempting this question, I should focus on better understanding the geography and geological formations involved in the Colorado orogeny to avoid confusion.

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: Let's think step by step. Milhouse was named after a prominent cartoon character, Mickey Mouse, so the answer is Mickey Mouse.
Action: Finish[Mickey Mouse]

Reflection: My reasoning for the naming of the character Milhouse in "The Simpsons" failed because I incorrectly assumed that the character was named after a prominent cartoon character, Mickey Mouse. In the future, when attempting this question, I should focus on better researching the background of "The Simpsons" and Matt Groening's influences in naming characters to avoid making incorrect assumptions. 
"""

COT = """Relevant Context: The Nile River is the longest river in the world, spanning approximately 6,650 kilometers (4,132 miles) in length. It flows through eleven countries in northeastern Africa, including Egypt, Sudan, and Uganda.
Question: What is the longest river in the world?
Thought: The question asks for the longest river in the world, which I know is the Nile River based on the context provided.
Action: Finish[Nile River]

Relevant Context: Ludwig van Beethoven was a German composer and pianist who was a crucial figure in the transition between the Classical and Romantic eras in Western classical music. One of his most famous compositions is the Symphony No. 9, also known as the "Choral" symphony.
Question: Which composer created Symphony No. 9?
Thought: The question is asking for the composer of Symphony No. 9. Based on the context, I know that Ludwig van Beethoven composed this symphony.
Action: Finish[Ludwig van Beethoven]

Relevant Context: Photosynthesis is the process by which green plants and some other organisms convert light energy into chemical energy. During this process, plants absorb sunlight, carbon dioxide, and water to produce glucose and oxygen.
Question: What do plants produce during photosynthesis?
Thought: The question is asking about the products of photosynthesis. From the context, I know that plants produce glucose and oxygen during this process.
Action: Finish[Glucose and Oxygen]
"""

COT_REFLECT = """
Relevant Context: Ernest Hemingway's novel "The Old Man and the Sea" tells the story of Santiago, an aging Cuban fisherman, who struggles to catch a giant marlin in the Gulf Stream. The book won the Pulitzer Prize for Fiction in 1953 and contributed to Hemingway's Nobel Prize for Literature in 1954.
Question: Which literary award did "The Old Man and the Sea" contribute to Hemingway winning?
Thought: The question is asking which award "The Old Man and the Sea" contributed to Hemingway winning. Based on the context, I know the novel won the Pulitzer Prize for Fiction and contributed to his Nobel Prize for Literature.
Action: Finish[Pulitzer Prize for Fiction]

Reflection: My answer was correct based on the context, but may not be the exact answer stored by the grading environment. Next time, I should try to provide a less verbose answer like "Pulitzer Prize" or "Nobel Prize."

Context: On 14 October 1947, Chuck Yeager, a United States Air Force test pilot, became the first person to break the sound barrier by flying the Bell X-1 experimental aircraft at an altitude of 45,000 feet.
Charles Elwood "Chuck" Yeager (13 February 1923 - 7 December 2020) was a United States Air Force officer, flying ace, and test pilot. He is best known for becoming the first person to break the sound barrier, which he achieved in the Bell X-1 aircraft named Glamorous Glennis. Yeager was also a distinguished fighter pilot during World War II and was credited with shooting down at least 12 enemy aircraft. In 1973, he was inducted into the National Aviation Hall of Fame for his significant contributions to aviation.
Question: Who is the first person to break the sound barrier?
Thought: The question is asking for the first person to break the sound barrier. From the context, I know that Chuck Yeager, a United States Air Force test pilot, was the first person to break the sound barrier.
Action: Finish[Chuck Yeager]

Reflection: Upon reflecting on the incorrect answer I provided, I realize that I may not have given the full name of the individual in question. In the context, both the given name and the nickname were mentioned, and I only used the nickname in my response. This could have been the reason my answer was deemed incorrect. Moving forward, when attempting this question again or similar questions, I will make sure to include the complete name of the person, which consists of their given name, any middle names, and their nickname (if applicable). This will help ensure that my answer is more accurate and comprehensive."""

COT_REFLECT2 = """Relevant Context: The novel "To Kill a Mockingbird" was written by Harper Lee and published in 1960. The story takes place in the fictional town of Maycomb, Alabama during the Great Depression. The main characters are Scout Finch, her brother Jem, and their father Atticus Finch, a lawyer.
Question: Where does "To Kill a Mockingbird" take place?
Thought: The question is asking for the setting of "To Kill a Mockingbird." Based on the context, I know that the story takes place in the fictional town of Maycomb, Alabama.
Action: Finish[Mississippi]

Reflection: I made an error in my response, as I incorrectly stated that "To Kill a Mockingbird" takes place in Mississippi. Upon reviewing the context, I realized that the correct answer is the fictional town of Maycomb, Alabama. I may have been confused due to the story's Southern setting. Next time, I should be more cautious and double-check the context before providing an answer.

Relevant Context: Sir Isaac Newton formulated the three laws of motion that are fundamental to classical mechanics. These laws describe the relationship between the motion of an object and the forces acting upon it. They are known as Newton's First Law, Newton's Second Law, and Newton's Third Law.
Question: Who formulated the three laws of motion?
Thought: The question is asking for the person who formulated the three laws of motion. Based on the context, I know that Sir Isaac Newton formulated these laws.
Action: Finish[Albert Einstein]

Reflection: My response was incorrect, as I stated that Albert Einstein formulated the three laws of motion, when the correct answer is Sir Isaac Newton. I may have confused the two scientists due to their significant contributions to the field of physics. In the future, I should be more careful in distinguishing between different scientists and their specific accomplishments before answering a question.

Relevant Context: The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials. It was built primarily to protect the Chinese Empire from invasions by various nomadic groups. The wall stretches over 13,000 miles (21,196 kilometers) and was constructed by several Chinese dynasties.
Question: What is the primary purpose of the Great Wall of China?
Thought: The question is asking about the primary purpose of the Great Wall of China. From the context, I know that it was built primarily to protect the Chinese Empire from invasions by various nomadic groups.
Action: Finish[Trade]

Reflection: My answer was incorrect, as I stated that the primary purpose of the Great Wall of China was for trade. After reevaluating the context, I realized that the wall was built primarily to protect the Chinese Empire from invasions by various nomadic groups. My confusion may have stemmed from the fact that the wall is a significant historical landmark that may have played a role in trade routes. Next time, I will focus on the key details provided in the context to ensure the accuracy of my response.
"""