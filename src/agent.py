from langchain.agents import ZeroShotAgent, AgentExecutor
from typing import List
from langchain_core.tools import BaseTool
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOllama
import logging
from langchain.chains import LLMChain

from src.fetch import get_programs_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3", temperature=0.3)
        self.user_sessions = {}

    def _prepare_tools(self) -> List[BaseTool]:
        """Подготовка инструментов с явными описаниями"""
        return [
            get_programs_info
        ]

    def _create_agent_executor(self, tools: List[BaseTool]):
        """Создание агента с ZeroShot подходом"""
        prefix = """Ты - помощник абитуриента. Ты помогаешь разобраться, какая из двух магистерских программ ему подходит.
        Программы:
        1. 'ai'
        2. 'ai_product'

        Для получения информации о программах используй инструмент get_programs_info.
        """
        
        suffix = """Вопрос: {input}

Текущие данные: {agent_scratchpad}
"""

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["agent_scratchpad"]
        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=ConversationBufferWindowMemory(k=3),
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15
        )

    def invoke(self, user_id: str, question: str) -> str:
        """Обработка запроса пользователя"""
        user_id = str(user_id)
        if user_id not in self.user_sessions:
            tools = self._prepare_tools()
            self.user_sessions[user_id] = self._create_agent_executor(tools)
        
        executor = self.user_sessions[user_id]
        result = executor.invoke({
            "input": question
        })
        output = result['output']
        return output