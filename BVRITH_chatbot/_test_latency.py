import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import logging
logging.basicConfig(level=logging.WARNING)
from dotenv import load_dotenv
load_dotenv(override=True)

from vector_store import get_vector_store
from chatbot import CollegeChatbot

vs, _ = get_vector_store()
bot = CollegeChatbot(vs)

questions = ['What B.Tech branches does BVRIT offer?', 'What is the fee structure?']
for q in questions:
    t = time.time()
    result = bot.answer_question(q)
    elapsed = time.time() - t
    ans = result.get('answer', '')
    print('Q:', q)
    print('Time:', round(elapsed, 2), 's')
    print('A:', ans[:400])
    print()
