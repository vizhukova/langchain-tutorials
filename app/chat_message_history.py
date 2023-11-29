import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
history = ChatMessageHistory()
history.add_ai_message("hi!")
history.add_user_message("what is the capital of france?")
print(history.messages)
# [AIMessage(content='hi!'),
#  HumanMessage(content='what is the capital of france?')]
ai_response = chat(history.messages)
print(ai_response)
# AIMessage(content='The capital of France is Paris.')
history.add_ai_message(ai_response.content)
print(history.messages)
# [AIMessage(content='hi!'),
#  HumanMessage(content='what is the capital of france?'),
#  AIMessage(content='The capital of France is Paris.')]

# Chains Chat Message History
# https://python.langchain.com/en/latest/modules/chains/how_to_guides.html
print('/////////////////////////////')

lm = ChatOpenAI()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(
    llm=lm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

conversation({"question": "hi"})

print('////////////////////////////')
# Simple Sequential Chains
template = """Your job is to come up with a classic dish from the area that the users suggests.
% USER LOCATION
{user_location}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_location"], template=template)

# Holds my 'location' chain
location_chain = LLMChain(llm=lm, prompt=prompt_template)

template = """Given a meal, give a short and simple recipe on how to make that dish at home.
% MEAL
{user_meal}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)

# Holds my 'meal' chain
meal_chain = LLMChain(llm=lm, prompt=prompt_template)

overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)
review = overall_chain.run("Rome")