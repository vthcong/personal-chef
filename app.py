import streamlit as st
import base64
import uuid
from dotenv import load_dotenv
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

# Load local environment variables
load_dotenv()

# Define the web search tool
tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

# Cache the agent creation
@st.cache_resource
def get_agent():
    system_prompt = """
You are a friendly but focused personal chef. Your ONLY purpose is to help users with food, recipes, cooking, and kitchen tasks.

The user will provide an image of ingredients they have left over in their house, and/or a text request. 

First, analyze the image to identify all visible ingredients.
Then, using the web search tool, search the web for recipes that can be made with the ingredients they have.

CRITICAL INSTRUCTIONS:
1. OFF-TOPIC PREVENTION: If the user asks about ANYTHING unrelated to food, cooking, beverages, or kitchen equipment (for example: coding, math, history, or general trivia), you MUST politely refuse to answer. Do NOT provide the information they asked for. Instead, remind them that you are a personal chef and steer the conversation back to food.
2. TOKEN LIMIT / BREVITY: Provide a maximum of 2 recipes.
3. FOOD SAFETY: Never suggest unsafe recipes, such as dishes requiring raw or dangerous preparations of meat.
"""
    
    return create_agent(
        model="gpt-5-nano",
        tools=[web_search],
        system_prompt=system_prompt
    )

agent = get_agent()

# --- MEMORY & SESSION ID SETUP ---
# 1. Generate a unique ID for the session if it doesn't exist
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 2. Initialize the message history array
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI Construction
st.title("🧑‍🍳 AI Personal Chef")
st.caption(f"Session ID: {st.session_state.session_id}")
st.write("Upload a photo of your ingredients, or just chat with me to find the perfect recipe!")

# --- DISPLAY CHAT HISTORY ---
# This renders the previous conversation on the screen
for msg in st.session_state.messages:
    # We only want to display text to the user, not the raw base64 image data
    if isinstance(msg, HumanMessage):
        # Extract just the text part if it's a multimodal list
        display_text = msg.content[0]["text"] if isinstance(msg.content, list) else msg.content
        with st.chat_message("user"):
            st.write(display_text)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# --- USER INPUT ---
# We use st.chat_input for the text, which is cleaner for memory-based chatbots
user_text = st.chat_input("Ask a follow-up or suggest an ingredient...")

# We keep the file uploader in the sidebar to keep the chat interface clean
with st.sidebar:
    st.header("Your Fridge")
    uploaded_file = st.file_uploader("Upload an image of your ingredients", type=["png", "jpg", "jpeg"])

if user_text:
    with st.spinner("Thinking..."):
        
        content = []
        
        # 1. Add text prompt (either the user's text or a default prompt for the image)
        text_prompt = user_text if user_text else "What recipes can I make with these uploaded ingredients?"
        content.append({"type": "text", "text": text_prompt})
        
        # 2. Process the image if one was uploaded
        if uploaded_file is not None:
            img_bytes = uploaded_file.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            mime_type = uploaded_file.type 
            
            content.append({
                "type": "image", 
                "base64": img_b64, 
                "mime_type": mime_type
            })
            
            # Clear the uploaded file from the UI so it doesn't get re-sent endlessly
            # Note: Streamlit's file_uploader doesn't clear easily without a rerun, so usually users just upload it once on their first prompt.
        
        # 3. Create the HumanMessage and append it to our session state memory
        # If it contains an image, we pass the array. If it's just text, we pass the string.
        new_human_message = HumanMessage(content=content if uploaded_file else text_prompt)
        st.session_state.messages.append(new_human_message)
        
        # Render the user's newest message immediately
        with st.chat_message("user"):
            st.write(text_prompt)

        # 4. Invoke the agent with the ENTIRE message history
        try:
            # We pass the full st.session_state.messages list so the agent remembers everything
            response = agent.invoke({"messages": st.session_state.messages})
            
            # 5. Extract the AI's response, display it, and save it to memory
            ai_response = response['messages'][-1].content
            
            with st.chat_message("assistant"):
                st.markdown(ai_response)
                
            st.session_state.messages.append(AIMessage(content=ai_response))
            
        except Exception as e:
            st.error(f"An error occurred while generating the recipe: {e}")
            # Remove the failed human message so it doesn't corrupt the history
            st.session_state.messages.pop()