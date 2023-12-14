#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install google-api-python-client


# In[2]:


get_ipython().system('pip install -q openai langchain py_expression_eval google-api-python-client')


# In[3]:


from openai import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool
from py_expression_eval import Parser
import re, time, os


# In[9]:


client = OpenAI(api_key='your open ai api key')
os.environ["GOOGLE_CSE_ID"] = "your google cse id"
os.environ["GOOGLE_API_KEY"] = "your google api key"


# In[10]:


#Calculator
parser = Parser()
def calculator(str):
    return parser.parse(str).evaluate({})

#Google search engine
search = GoogleSearchAPIWrapper()
goog = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run
    )

def search(str):
    return goog(str)


# In[11]:


System_prompt = """
Answer the following questions and obey the following commands as best you can.

You have access to the following tools:

Search: Search: useful for when you need to answer questions about current events. You should ask targeted questions.
Calculator: Useful for when you need to answer questions about math. Use python code, eg: 2 + 2
Response To Human: When you need to respond to the human you are talking to.

You will receive a message from the human, then you should start a loop and do one of two things

Option 1: You use a tool to answer the question.
For this, you should use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [Search, Calculator]
Action Input: "the input to the action, to be sent to the tool"

After this, the human will respond with an observation, and you will continue.

Option 2: You respond to the human.
For this, you should use the following format:
Action: Response To Human
Action Input: "your response to the human, summarizing what you did and what you learned"

Begin!
"""


# In[18]:


def Stream_agent(prompt):
    messages = [
        { "role": "system", "content": System_prompt },
        { "role": "user", "content": prompt },
    ]
    def extract_action_and_input(text):
          action_pattern = r"Action: (.+?)\n"
          input_pattern = r"Action Input: \"(.+?)\""
          action = re.findall(action_pattern, text)
          action_input = re.findall(input_pattern, text)
          return action, action_input
    while True:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0,
            max_tokens=1000,
            top_p=1,)
        response_text = response.choices[0].message.content
        print(response_text)
        #To prevent the Rate Limit error for free-tier users, we need to decrease the number of requests/minute.
        time.sleep(20)
        action, action_input = extract_action_and_input(response_text)
        if action[-1] == "Search":
            tool = search
        elif action[-1] == "Calculator":
            tool = calculator
        elif action[-1] == "Response To Human":
            print(f"Response: {action_input[-1]}")
            break
        observation = tool(action_input[-1])
        print("Observation: ", observation)
        messages.extend([
            { "role": "system", "content": response_text },
            { "role": "user", "content": f"Observation: {observation}" },
            ])


# In[20]:


Stream_agent("who is current CEO of OpenAI")


# In[21]:


Stream_agent("How is the weather today in Deggendorf, Bavaria, Germany?")


# In[22]:


Stream_agent("What are the latest trends in digital forensics practice?")


# In[24]:


Stream_agent("What are the newest trends in digital forensics?")


# In[25]:


Stream_agent("Propose tools I can use for device forensics on Internet of Things (IoT) devices")


# In[26]:


Stream_agent("Can you visit this website: https://www.documentcloud.org/documents/6668313-FTI-Report-into-Jeff-Bezos-Phone-Hack and provide a summary of the PDF contents")


# In[27]:


Stream_agent("Which tools were used to realize the Jeff Bezos Hack?")


# In[28]:


Stream_agent("Provide a list of tools for IoT forensics and compare their functionalities")


# In[29]:


Stream_agent("Do you know Luwig Englbrecht, what does he do?")


# In[ ]:




