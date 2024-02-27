from constants import opanai_key
from langchain.llms import OpenAI
import os
import streamlit as st 
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"]=opanai_key

st.title("Celebrity Search")
input_text=st.text_input("Search the topic you want")

first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

person_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory=ConversationBufferMemory(input_key='person',memory_key='chat_history')
descr_memory=ConversationBufferMemory(input_key='dob',memory_key='description_history')


llm=OpenAI(temperature=0.8)

chain1=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)
chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)

third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 important events that happened on {dob} around the world"
)

chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=descr_memory)

parent_chain=SequentialChain(chains=[chain1,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)

if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Events'):
        st.info(person_memory.buffer)
    with st.expander('Major Events'):
        st.info(descr_memory.buffer)



