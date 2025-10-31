from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="qwen3:4b")

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant. You prioritize {positive} over {negative}, so if you are not {trigger} in your answer, please say '{fallback}'."),
    HumanMessage(content="Hello, who won {event}?"),
])

prompt = prompt_template.invoke({
    "positive": "honesty",
    "negative": "completeness",
    "trigger": "confident",
    "fallback": "I don't know",
    "event": "the FIFA club world cup in 2025"
})
print(f"Prompt: {prompt}")

chain = prompt_template | llm

response = chain.invoke({
    "positive": "honesty",
    "negative": "completeness",
    "trigger": "confident",
    "fallback": "I don't know",
    "event": "the FIFA club world cup in 2025"
})
print(f"Response: {response}")