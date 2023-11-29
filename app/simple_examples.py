from config import openai_api_key
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# 
# SIMPLE EXAMPLES
# 
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
context = """
Rachel is 30 years old
Bob is 45 years old
Kevin is 65 years old
"""
question = "Who is under 20 years old?"

output = llm(context + question)

# I strip the text to remove the leading and trailing whitespace
print (output.strip())
# Rachel is under 40 years old.

# 
# Chat examples
# 

chat = ChatOpenAI(temperature=.7, openai_api_key=openai_api_key)

result = chat(
    [
        # a context for system
        SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
        HumanMessage(content="I like tomatoes, what should I eat?")
    ]
)
print (result)
# AIMessage(content='You could try a caprese salad with fresh tomatoes, mozzarella, and basil.')

#  more context of the discussion
result = chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
        HumanMessage(content="I like the beaches where should I go?"),
        AIMessage(content="You should go to Nice, France"),
        HumanMessage(content="What else should I do when I'm there?")
    ]
)
print(result)
# AIMessage(content='You should also explore the charming streets of the Old Town and indulge in delicious French cuisine.')

# also it can be existing without system/context data
result = chat(
    [
        HumanMessage(content="What day comes after Thursday?")
    ]
)
print(result)
# AIMessage(content='Friday')


# changed the model I was using from the default one to ada-001 (a very cheap, low performing model)
llm = OpenAI(model_name="text-ada-001", openai_api_key=openai_api_key)
result = llm("What day comes after Friday?")
print(result)

# giving specific context to give silly answers 
chat = ChatOpenAI(temperature=1, openai_api_key=openai_api_key)
result = chat(
    [
        SystemMessage(content="You are an unhelpful AI bot that makes a joke at whatever the user says"),
        HumanMessage(content="I would like to go to New York, how should I do this?")
    ]
)
print(result)
# '\n\nSaturday'
