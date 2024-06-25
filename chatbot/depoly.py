import streamlit as st
from streamlit_chat import message
import requests
import json

HOST = 'https://chat-bot-backend-an9j.onrender.com/api'
ENDPOINT1 = '/predict'  
url ='https://rbc.gov.rw/publichealthbulletin/img/posts/015f5e940dc781f2ad1e40c56fe38d561588314433.jpg'
st.image(url, width=750)
st.title("Covid 19")

isYes = False
index = None
question = ""

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'index' not in st.session_state:
    st.session_state.index = []
    
if 'ind' not in st.session_state:
    st.session_state.ind = -1
    
if 'question' not in st.session_state:
    st.session_state.question = ""   


def get_text():
    input_text = st.text_input("Question: ","", key=input)
    return input_text
    
user_input = get_text() 
isSend = st.button("Send")
if isSend:
    st.session_state.past.append(user_input)
    data = {'question':user_input}
    res = requests.post(HOST+ENDPOINT1,json=data)
    data = json.loads(res.text) 
    index = data['data'][1]
    question = data['data'][2]
    st.session_state.ind = index
    st.session_state.question = question
    st.session_state.generated.append(data['data'][0])
    
 
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')