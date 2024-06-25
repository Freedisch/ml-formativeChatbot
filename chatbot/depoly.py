import streamlit as st
from streamlit_chat import message
import requests
import json


# Constants 
# HOST = 'http://127.0.0.1:8080/api'  # HOST SERVER URL
HOST = 'https://chat-bot-backend-an9j.onrender.com/api'
####### ENDPOINT of different Api ####################333
ENDPOINT1 = '/predict'  
# ENDPOINT2 = '/train'
# ENDPOINT3 = '/addQuestion'

# st.image('breastcancer.jpg', width=750)
url ='https://images.theconversation.com/files/487963/original/file-20221004-26-x590oy.jpg?ixlib=rb-4.1.0&rect=0%2C237%2C4288%2C2144&q=45&auto=format&w=1356&h=668&fit=crop'

st.image(url, width=750)
# st.header("Breast Cancer Chat-Bot")
st.title("Breast Cancer Chat-Bot")

isYes = False
index = None
question = ""


## store session_state variable to load after old session after clicking buttons or show different messages 
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
    """
        get user input in a text box and store in input_text variable
    """
    input_text = st.text_input("Question: ","", key=input)
    return input_text
    

## store user input
user_input = get_text() 
# _, btn1, btn2 = st.columns([4,0.5,0.5])
isSend = st.button("Send")
# isTrain = btn2.button("Train")


## check if train button is click or not
# if isTrain:
#     if len(st.session_state.index) > 0:
#         res = requests.get(HOST+ENDPOINT2)
#         data = json.loads(res.text)
#         message(data['msg'])


## check if send button is click or not
if isSend:
    st.session_state.past.append(user_input)
    data = {'question':user_input}
    ## hit the predict api and get the response
    res = requests.post(HOST+ENDPOINT1,json=data)
    data = json.loads(res.text) 
    index = data['data'][1]
    question = data['data'][2]
    st.session_state.ind = index
    st.session_state.question = question
    st.session_state.generated.append(data['data'][0])
    
 
# if len(st.session_state.generated) > 0 and st.session_state.generated[-1] != "I Donot know about it":
#     col1, col2, col3 = st.columns([4,0.5,0.5])
#     place3 = col3.empty()
#     place2 = col2.empty()
#     place1 = col1.empty()
#     place1.text("Are you Satisfied with the Response ?")
#     isYes = place2.button("YES")
#     isNo = place3.button("NO")
#     if isYes:
#         place1.empty()
#         place2.empty()
#         place3.empty()
#         st.session_state.index.append(st.session_state.ind)
#         data = {'question': st.session_state.question,'index':st.session_state.ind}
#         res = requests.post(HOST+ENDPOINT3,json=data)
#         data = json.loads(res.text)
#     if isNo:
#         place1.empty()
#         place2.empty()
#         place3.empty()

### show all message that user ask and the response that the user get
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')