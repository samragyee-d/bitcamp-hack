# state.py
from collections import deque

chat_history = []

def clear_chat_history():
    chat_history.clear()
    
recording_flag = {"status": False}  # Global recording flag
