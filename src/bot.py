from time import sleep

from origamibot import OrigamiBot as Bot
from origamibot.listener import Listener
from src.agent import LLMAgent
from src.depends import TG_BOT_TOKEN

class BotsCommands:
    def __init__(self, bot: Bot):  # Can initialize however you like
        self.bot = bot

    def start(self, message):   # /start command
        self.bot.send_message(
            message.chat.id,
            'Привет!')

agent = LLMAgent()

class MessageListener(Listener):  # Event listener must inherit Listener
    def __init__(self, bot):
        self.bot = bot

    def on_message(self, message):
        print(f"{message.chat.id}> {message.text}")
        reply = ""
        if message.text != '/start':
            reply = agent.invoke(message.chat.id, message.text)
        self.bot.send_message(message.chat.id, reply)

    # def on_command_failure(self, message, err=None):
    #     if err is None:
    #         self.bot.send_message(message.chat.id,
    #                               'Command failed to bind arguments!')
    #     else:
    #         self.bot.send_message(message.chat.id,
    #                               f'Error in command:\n{err}')


if __name__ == '__main__':
    bot = Bot(TG_BOT_TOKEN)
    bot.add_listener(MessageListener(bot))
    bot.add_commands(BotsCommands(bot))
    bot.start()
    while True:
        sleep(1)