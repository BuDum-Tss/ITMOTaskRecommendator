from langchain.agents import ZeroShotAgent, AgentExecutor
from typing import List, Dict, Any
from langchain_core.tools import BaseTool, Tool
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
        """Явное описание инструментов с чёткими инструкциями"""
        return [
            Tool(
                name="get_programs_info",
                func=get_programs_info,
                description="""Достаёт информацию о программах. 
                Входные параметры: None. 
                Возвращает словарь с данными о программах 'ai' и 'ai_product'"""
            )
        ]

    def _create_agent_executor(self, tools: List[BaseTool]) -> AgentExecutor:
        """Улучшенный prompt с примерами"""
        prefix = """Ты - помощник абитуриента. Ты помогаешь выбрать между двумя программами:
        1. 'ai' - Искусственный интеллект
        2. 'ai_product' - Управление ИИ-продуктами

        Инструкции:
        - Сначала собери информацию с помощью get_programs_info
        - Анализируй данные и давай чёткие рекомендации
        - Отвечай на русском языке
        - Если не знаешь ответа - скажи об этом

        Пример работы:
        Пользователь: Какая программа лучше для ML-инженера?
        Ты: [действие] Запускаю get_programs_info для сравнения...
        [анализ данных] Программа "Искусственный интеллект" лучше подходит, так как...
        """
        
        suffix = """Начни!

        Вопрос: {input}
        {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "agent_scratchpad"]
        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        agent = ZeroShotAgent(
            llm_chain=llm_chain, 
            tools=tools,
            handle_parsing_errors=True
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=ConversationBufferWindowMemory(k=3),
            verbose=True,
            max_iterations=5,  # Уменьшил для предотвращения циклов
            early_stopping_method="generate"
        )

    def invoke(self, user_id: str, question: str) -> str:
        """Добавлена проверка входных данных"""
        if not question or not isinstance(question, str):
            return "Пожалуйста, задайте вопрос о программах"
            
        user_id = str(user_id)
        if user_id not in self.user_sessions:
            tools = self._prepare_tools()
            self.user_sessions[user_id] = self._create_agent_executor(tools)
        
        try:
            result = self.user_sessions[user_id].invoke({"input": question})
            return result.get("output", "Не удалось получить ответ")
        except Exception as e:
            logger.error(f"Error in invoke: {e}")
            return "Произошла ошибка при обработке запроса"