# import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
AVALIABLE_COMPETIIIONS_JSON_PATH = os.getenv("AVALIABLE_COMPETIIIONS_JSON_PATH")