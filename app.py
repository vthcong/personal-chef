import streamlit as st
import base64
from dotenv import load_dotenv
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.messages import HumanMessage

# Load local environment variables (for testing locally)
load_dotenv()

# Define the web search tool
tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

# Cache the agent creation so it doesn't reload on every UI interaction
@st.cache_resource
def get_agent():
    system_prompt = """
    You are a friendly but focused personal chef. Your ONLY purpose is to help users with food, recipes, cooking, and kitchen tasks.

    The user will provide an image of ingredients they have left over in their house, and/or a text request. 

    First, analyze the image to identify all visible ingredients.
    Then, using the web search tool, search the web for recipes that can be made with the ingredients they have.

    CRITICAL INSTRUCTIONS:
    1. OFF-TOPIC PREVENTION: If the user asks about ANYTHING unrelated to food, cooking, beverages, or kitchen equipment (for example: coding, math, history, or general trivia), you MUST politely refuse to answer. Do NOT provide the information they asked for. Instead, remind them that you are a personal chef and steer the conversation back to food.
    2. TOKEN LIMIT / BREVITY: Keep your responses as concise as possible. Provide a maximum of 2 recipes.
    3. FOOD SAFETY: Never suggest unsafe recipes, such as dishes requiring raw or dangerous preparations of meat.
    """
    
    return create_agent(
        model="gpt-5-nano",
        tools=[web_search],
        system_prompt=system_prompt
    )

agent = get_agent()

# Streamlit UI Construction
st.title("🧑‍🍳 AI Personal Chef")
st.write("Upload a photo of your fridge, pantry, or leftover ingredients, and I'll find the perfect recipe!")

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image of your ingredients", type=["png", "jpg", "jpeg"])

# Optional text input for specific cravings
user_text = st.text_input("Any specific cravings? (e.g., 'Make it spicy', 'I want a dessert')")

if st.button("Find Recipes"):
    if uploaded_file or user_text:
        with st.spinner("Analyzing ingredients and searching for recipes..."):
            
            # 1. Build the content array for the HumanMessage
            content = []
            
            # Add text prompt
            text_prompt = user_text if user_text else "What recipes can I make with these ingredients?"
            content.append({"type": "text", "text": text_prompt})
            
            # 2. Process the image if one was uploaded
            if uploaded_file is not None:
                # Read the file directly into bytes
                img_bytes = uploaded_file.getvalue()
                # Base64 encode the bytes
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                # Extract mime type dynamically (e.g., 'image/png' or 'image/jpeg')
                mime_type = uploaded_file.type 
                
                # Append the image dictionary just like in the notebook
                content.append({
                    "type": "image", 
                    "base64": img_b64, 
                    "mime_type": mime_type
                })
            
            # 3. Construct the multimodal message
            multimodal_question = HumanMessage(content=content)
            
            # 4. Invoke the agent
            try:
                response = agent.invoke({"messages": [multimodal_question]})
                # Display the model's response
                st.markdown(response['messages'][-1].content)
            except Exception as e:
                st.error(f"An error occurred while generating the recipe: {e}")
    else:
        st.warning("Please upload an image or type a request to get started.")