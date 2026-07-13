"""
Core chatbot logic for the College FAQ Chatbot.

Handles the QA chain, conversation history management,
LLM invocation, and answer generation with citations.
Extended with OpenAI/OpenRouter Function Calling for
fee_calculator, date_checker, and percentage_calculator tools.
"""

import re
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

from utils import get_config, format_time, Timer
from vector_store import get_embeddings, retrieve_documents, TOP_K
from prompts import ANSWER_PROMPT
from tools import TOOLS, execute_tool
from memory import MemoryManager

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Social / Conversational Pattern Detection
# ──────────────────────────────────────────────

# Greeting patterns
GREETING_PATTERNS = [
    r"^(hi|hello|hey|heyy|howdy)\b",
    r"^good\s*(morning|afternoon|evening|day)\b",
    r"^what'?s\s*up\b",
    r"^sup\b",
    r"^namaste\b",
    r"^yo\b",
]

GREETING_RESPONSE = (
    "Hello! 👋 Welcome to BVRIT. I'm your college assistant and I'm here to help you with "
    "admissions, departments, placements, fees, campus facilities, and more. "
    "What would you like to know?"
)

# Gratitude patterns
THANKS_PATTERNS = [
    r"^thanks?\b",
    r"^thank\s*you\b",
    r"^thx\b",
    r"^ty\b",
    r"^appreciate\s*it\b",
    r"^much\s*appreciated\b",
]

THANKS_RESPONSE = (
    "You're welcome! 😊 Happy to help. Let me know if you have any other questions."
)

# Farewell patterns
FAREWELL_PATTERNS = [
    r"^(bye|goodbye|see\s*ya|see\s*you|take\s*care|cya|gottago)\b",
    r"^good\s*night\b",
    r"^catch\s*you\s*later\b",
]

FAREWELL_RESPONSE = (
    "Goodbye! 👋 Have a great day, and feel free to come back anytime "
    "if you have more questions."
)

# Introductory phrases for knowledge responses
KNOWLEDGE_INTROS = [
    "Sure! Here's what I found:",
    "Great question! Based on the college information...",
    "I'd be happy to help with that! Here's what I know:",
    "Here's what the knowledge base says:",
    "Of course! Let me look that up for you:",
]


class CollegeChatbot:
    """
    Main chatbot class for the College FAQ Chatbot application.

    Manages the LLM, vector store retrieval, conversation history,
    conversational social detection, tool execution (function calling),
    and the answer generation pipeline.
    """

    def __init__(self, vector_store: Chroma, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize the chatbot with a vector store and optional memory system.

        Args:
            vector_store (Chroma): The ChromaDB vector store instance (knowledge base).
            memory_manager (Optional[MemoryManager]): The memory system for user memories.
                                                     If None, memory features are disabled.
        """
        self.vector_store = vector_store
        self.memory_manager = memory_manager or MemoryManager()
        self.chat_history: List[Dict[str, str]] = []
        self.llm = self._initialize_llm()
        self._intro_index = 0
        # Track the last tool debug info
        self.last_tool_debug: Dict[str, Any] = {}
        # Track current user_id and session_id for memory operations
        self.current_user_id: str = ""
        self.current_session_id: str = ""

    def _initialize_llm(self) -> ChatOpenAI:
        """
        Initialize the language model via OpenRouter.

        Returns:
            ChatOpenAI: Configured LLM instance.

        Raises:
            ValueError: If the API key is missing.
        """
        config = get_config()
        api_key = config.get("OPENROUTER_API_KEY")

        if not api_key or api_key == "your-api-key-here":
            raise ValueError(
                "OpenRouter API key is missing. "
                "Please set OPENROUTER_API_KEY in your .env file."
            )

        llm = ChatOpenAI(
            model=config.get("OPENROUTER_MODEL", "gpt-4o-mini"),
            openai_api_key=api_key,
            openai_api_base=config.get("OPENROUTER_BASE_URL"),
            temperature=0.1,
            max_tokens=1024,
        )
        return llm

    def _detect_social_intent(self, text: str) -> Optional[str]:
        """
        Detect if the user's message is a greeting, thanks, or farewell.

        Args:
            text (str): The user's input text.

        Returns:
            Optional[str]: A predefined social response, or None if it's a
                           knowledge-based question.
        """
        cleaned = text.strip().lower()

        # Check greetings
        for pattern in GREETING_PATTERNS:
            if re.search(pattern, cleaned):
                return GREETING_RESPONSE

        # Check thanks
        for pattern in THANKS_PATTERNS:
            if re.search(pattern, cleaned):
                return THANKS_RESPONSE

        # Check farewells
        for pattern in FAREWELL_PATTERNS:
            if re.search(pattern, cleaned):
                return FAREWELL_RESPONSE

        return None

    def _get_intro_phrase(self) -> str:
        """
        Cycle through introductory phrases for variety.

        Returns:
            str: A friendly introductory phrase.
        """
        phrase = KNOWLEDGE_INTROS[self._intro_index % len(KNOWLEDGE_INTROS)]
        self._intro_index += 1
        return phrase

    def answer_question(
        self, question: str
    ) -> Dict:
        """
        Answer a user question using RAG (Retrieval-Augmented Generation)
        and/or OpenAI Function Calling tools, with integrated user memory.

        Routing paths:
        A. Normal conversation (greeting/thanks/farewell) → No tool, No RAG
        B. RAG only → Retrieve from ChromaDB → Generate answer
        C. RAG + Tool → Retrieve from KB → Execute tool → Generate final answer
        D. Tool only → Execute tool directly → Generate final answer

        Memory Pipeline:
        1. BEFORE answer: Retrieve user memories → Inject into system prompt
        2. AFTER answer: Extract new memories from conversation → Store in memory ChromaDB

        Args:
            question (str): The user's question.

        Returns:
            Dict: Response containing:
                - answer (str): Generated answer text
                - retrieved_chunks (List[Document]): Retrieved documents
                - confidence (float): Maximum relevance score
                - response_time (float): Time taken to generate response
                - chunk_count (int): Number of retrieved chunks
                - tool_debug (Dict): Tool execution debug info
                - routing (str): The routing path used
                - memories_used (int): Number of memories retrieved
                - memories_stored (int): Number of new memories stored
        """
        # Reset tool debug
        self.last_tool_debug = {}

        # ── Check for "clear my data" command ──
        if self.memory_manager.detect_clear_data_command(question):
            return self._handle_clear_data(question)

        # ── Step 0: Check for social intent (No tool, No RAG) ──
        social_response = self._detect_social_intent(question)
        if social_response:
            self._update_history(question, social_response)
            logger.info("Routing: ✓ Conversation")
            self.last_tool_debug = {"routing": "Conversation"}
            # Still extract memories from conversation context
            stored = self.memory_manager.extract_and_store_memories(
                user_id=self.current_user_id,
                user_message=question,
                assistant_response=social_response,
                session_id=self.current_session_id,
            )
            return {
                "answer": social_response,
                "retrieved_chunks": [],
                "confidence": 1.0,
                "response_time": "0.01s",
                "chunk_count": 0,
                "tool_debug": self.last_tool_debug,
                "routing": "Conversation",
                "memories_used": 0,
                "memories_stored": stored,
            }

        # ── RAG + Tool Pipeline (with Memory) ──
        with Timer() as timer:
            # Step 1: Rewrite query to improve retrieval
            expanded_query = self._rewrite_query(question)

            # Step 2: Retrieve relevant documents using expanded query
            retrieved_docs, max_score = retrieve_documents(
                self.vector_store, expanded_query, k=TOP_K
            )

            # Step 3: Format context
            context = self._format_context(retrieved_docs)

            # Step 4: Retrieve user memories and inject into prompt
            memory_context = self.memory_manager.retrieve_and_inject_memories(
                user_id=self.current_user_id,
                query=question,
                k=5,
            )

            # Step 5: Build messages with tools (now includes memory context)
            messages = self._build_tool_messages(question, context, memory_context)

            # Step 6: First LLM call — may return tool_calls or direct answer
            try:
                response = self.llm.invoke(messages, tools=TOOLS)
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                answer = (
                    "I encountered an error while processing your question. "
                    "Please try again later."
                )
                self.last_tool_debug = {"routing": "RAG (LLM error)"}
                self._update_history(question, answer)
                response_time = format_time(timer.duration)
                return {
                    "answer": answer,
                    "retrieved_chunks": retrieved_docs,
                    "confidence": round(max_score, 4),
                    "response_time": response_time,
                    "chunk_count": len(retrieved_docs),
                    "tool_debug": self.last_tool_debug,
                    "routing": "RAG (LLM error)",
                }

            # Step 7: Check for tool calls
            tool_calls = getattr(response, "tool_calls", None)

            if tool_calls:
                # ── Tool execution loop (OpenAI Function Calling style) ──
                tool_results = []
                for tc in tool_calls:
                    # Handle different response formats from LangChain
                    if hasattr(tc, "name"):
                        tool_name = tc.name
                        raw_args = tc.args if hasattr(tc, "args") else "{}"
                    elif hasattr(tc, "function"):
                        tool_name = tc.function.name if hasattr(tc.function, "name") else ""
                        raw_args = tc.function.arguments if hasattr(tc.function, "arguments") else "{}"
                    else:
                        tool_name = tc.get("name") or tc.get("function", {}).get("name", "")
                        raw_args = tc.get("args") or tc.get("function", {}).get("arguments", "{}")

                    try:
                        if isinstance(raw_args, str):
                            tool_args = json.loads(raw_args)
                        elif isinstance(raw_args, dict):
                            tool_args = raw_args
                        else:
                            tool_args = {}
                    except (json.JSONDecodeError, TypeError):
                        tool_args = {}

                    logger.info(f"Tool call detected: {tool_name} with args: {tool_args}")

                    # Execute the tool
                    tool_output = execute_tool(tool_name, tool_args)
                    tool_results.append({
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_output": tool_output,
                    })

                    # Store debug info
                    self.last_tool_debug = {
                        "routing": f"RAG + {tool_name}",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_output": tool_output,
                    }

                # Determine routing
                has_rag = bool(retrieved_docs and context != "No relevant context was found.")
                if has_rag:
                    routing = f"RAG + {tool_results[0]['tool_name']}" if tool_results else "RAG"
                else:
                    routing = f"Tool only ({tool_results[0]['tool_name']})" if tool_results else "RAG"

                logger.info(f"Routing: ✓ {routing}")
                self.last_tool_debug["routing"] = routing

                # Step 8: Send tool results back to the model for final answer
                final_messages = self._build_tool_result_messages(
                    question, context, tool_results, memory_context
                )

                try:
                    final_response = self.llm.invoke(final_messages)
                    answer = final_response.content
                except Exception as e:
                    logger.error(f"Final LLM invocation failed: {e}")
                    # Fallback: use tool output directly
                    answer_parts = [tr["tool_output"] for tr in tool_results]
                    answer = "\n\n".join(answer_parts)
            else:
                # No tool calls — RAG only
                answer = response.content
                routing = "RAG"
                logger.info(f"Routing: ✓ {routing}")
                self.last_tool_debug = {"routing": routing}

                # Step 9: Add conversational polish
            answer = self._polish_answer(answer, question)

            # Step 10: Update conversation history
            self._update_history(question, answer)

            # Step 11: Extract and store new memories from this exchange
            stored = self.memory_manager.extract_and_store_memories(
                user_id=self.current_user_id,
                user_message=question,
                assistant_response=answer,
                session_id=self.current_session_id,
            )

        response_time = format_time(timer.duration)

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_docs,
            "confidence": round(max_score, 4),
            "response_time": response_time,
            "chunk_count": len(retrieved_docs),
            "tool_debug": self.last_tool_debug,
            "routing": routing,
            "memories_used": len(memory_context.split("\n")) - 1 if memory_context else 0,
            "memories_stored": stored,
        }

    def _polish_answer(self, answer: str, question: str) -> str:
        """
        Add conversational polish to the RAG-generated answer.

        Prepends a friendly intro phrase and appends a friendly closing
        if the answer appears to be a substantive knowledge response.

        Args:
            answer (str): The raw answer from the LLM.
            question (str): The original user question.

        Returns:
            str: The polished conversational answer.
        """
        # Skip polishing for special responses
        if answer.startswith("I couldn't find") or answer.startswith("I encountered"):
            return answer

        # Skip if already has an intro-like start
        already_introed = any(
            answer.startswith(prefix) for prefix in [
                "Sure!", "Great question", "I'd be happy", "Here's what", "Of course"
            ]
        )

        if not already_introed:
            intro = self._get_intro_phrase()
            answer = f"{intro}\n\n{answer}"

        # Add a friendly closing if the answer doesn't already end with a question
        if not answer.rstrip().endswith("?") and not answer.rstrip().endswith("anything else"):
            closings = [
                "\n\nI hope that helps! Let me know if you'd like to know anything else.",
                "\n\nFeel free to ask if you have more questions about the college.",
                "\n\nIs there anything else I can help you with?",
                "\n\nLet me know if you need more information!",
            ]
            closing = closings[self._intro_index % len(closings)]
            answer = answer + closing

        return answer

    def _rewrite_query(self, question: str) -> str:
        """
        Rewrite the user's question to improve retrieval accuracy.

        Expands short/generic queries with college KB context terms
        so the embedding search matches the right document chunks.

        Args:
            question (str): The user's original question.

        Returns:
            str: The rewritten query string for better embedding search.
        """
        # Keyword expansion: maps common query topics to enriched search terms
        expansions = {
            "admission": "admission criteria eligibility requirements BVRIT",
            "placement": "placement companies recruitment packages BVRIT",
            "fee": "fee structure tuition cost expenses BVRIT",
            "course": "courses programs departments offered BVRIT",
            "facilities": "campus facilities infrastructure amenities BVRIT",
            "library": "library facilities resources BVRIT",
            "hostel": "hostel accommodation facilities BVRIT",
            "sports": "sports facilities athletics extracurricular BVRIT",
            "faculty": "faculty staff teaching professors BVRIT",
            "principal": "principal head of the college director BVRIT",
            "director": "director principal head of the college BVRIT",
            "chairman": "chairman founder president BVRIT",
            "secretary": "secretary correspondent administrative officer BVRIT",
            "hod": "head of department dean HOD BVRIT",
            "department": "department head dean faculty courses BVRIT",
            "contact": "contact phone email address reach BVRIT",
            "address": "address location reach campus BVRIT",
            "email": "email contact reach BVRIT",
            "phone": "phone contact number reach BVRIT",
            "website": "website site portal online BVRIT",
            "vision": "vision mission philosophy goals BVRIT",
            "mission": "mission vision philosophy goals BVRIT",
        }

        # Also detect "who is", "who are", "tell me about", "name the" patterns
        query_lower = question.lower()

        # For "who is X" or "who are X" — find documents mentioning X by name or role
        who_match = None
        for pattern in [r"\bwho\s+is\s+(.+?)$", r"\bwho\s+are\s+(.+?)$", r"\btell\s+me\s+about\s+(.+?)$"]:
            m = re.search(pattern, query_lower)
            if m:
                who_match = m.group(1).strip()
                break

        # Build expansion terms
        extra_terms = []

        # Check keyword expansions
        for keyword, expansion in expansions.items():
            if keyword in query_lower:
                extra_terms.append(expansion)

        # If asking about a specific person/role, add that as well
        if who_match:
            extra_terms.append(who_match)
            extra_terms.append(f"{who_match} details information BVRIT")

        if extra_terms:
            expanded = f"{question} {' '.join(extra_terms)}"
            logger.info(f"Expanded query: {expanded}")
            return expanded

        return question

    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a clear, readable context string.

        Each document is presented with its section heading and content.
        Designed to be easily parsed by the LLM.
        """
        context_parts = []
        seen_sections = set()
        for i, doc in enumerate(documents, 1):
            section = doc.metadata.get("section", "General")
            content = doc.page_content.strip()

            # Skip very short chunks
            if len(content) < 20:
                continue

            # Avoid duplicate sections
            if section in seen_sections:
                continue
            seen_sections.add(section)

            context_parts.append(
                f"---\nSection: {section}\n{content}\n"
            )

        if not context_parts:
            return "No relevant context was found."

        return "\n".join(context_parts)

    def _build_tool_messages(
        self, question: str, context: str, memory_context: str = ""
    ) -> List:
        """
        Build the message chain for the LLM with tool definitions.

        Includes system prompt, user memories, chat history, and the current
        question with context.

        Args:
            question (str): The user's question.
            context (str): Formatted context from retrieved documents.
            memory_context (str): Formatted user memories for prompt injection.

        Returns:
            List: List of LangChain message objects.
        """
        messages = []

        # Add system prompt
        system_content = ANSWER_PROMPT.messages[0].prompt.template

        # Inject user memories into the system prompt if available
        if memory_context:
            system_content = f"{system_content}\n\n{memory_context}"

        messages.append(SystemMessage(content=system_content))

        # Add chat history (last 6 messages for context)
        for entry in self.chat_history[-6:]:
            messages.append(HumanMessage(content=entry["question"]))
            messages.append(AIMessage(content=entry["answer"]))

        # Add current question with context
        human_message = ANSWER_PROMPT.messages[2].prompt.template.format(
            context=context, question=question
        )
        messages.append(HumanMessage(content=human_message))

        return messages

    def _build_tool_result_messages(
        self,
        question: str,
        context: str,
        tool_results: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> List:
        """
        Build the message chain after tool execution for the final LLM call.

        Args:
            question (str): The user's original question.
            context (str): The retrieved context.
            tool_results (List[Dict]): List of tool execution results.

        Returns:
            List: List of LangChain message objects for the final answer.
        """
        messages = []

        # System prompt
        system_content = ANSWER_PROMPT.messages[0].prompt.template

        # Inject user memories into the system prompt if available
        if memory_context:
            system_content = f"{system_content}\n\n{memory_context}"

        messages.append(SystemMessage(content=system_content))

        # Chat history
        for entry in self.chat_history[-6:]:
            messages.append(HumanMessage(content=entry["question"]))
            messages.append(AIMessage(content=entry["answer"]))

        # Original question with context
        human_message = ANSWER_PROMPT.messages[2].prompt.template.format(
            context=context, question=question
        )
        messages.append(HumanMessage(content=human_message))

        # Add tool results as a system message so the model can use them
        tool_summary = "Tool Execution Results:\n"
        for tr in tool_results:
            tool_summary += f"\nTool: {tr['tool_name']}\n"
            tool_summary += f"Arguments: {json.dumps(tr['tool_args'], indent=2)}\n"
            tool_summary += f"Output:\n{tr['tool_output']}\n"

        messages.append(SystemMessage(content=tool_summary))

        # Final instruction
        messages.append(
            HumanMessage(
                content=(
                    "Using the retrieved context and the tool execution results above, "
                    "provide a complete and helpful answer to the user's question. "
                    "Include the tool calculation results in your answer where relevant."
                )
            )
        )

        return messages

    def _update_history(self, question: str, answer: str) -> None:
        """
        Update the conversation history.

        Args:
            question (str): The user's question.
            answer (str): The generated answer.
        """
        self.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep only last 50 messages to prevent memory issues
        if len(self.chat_history) > 50:
            self.chat_history = self.chat_history[-50:]

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.chat_history = []
        logger.info("Conversation history cleared.")

    # ──────────────────────────────────────────────
    # Memory Integration Methods
    # ──────────────────────────────────────────────

    def set_user(self, user_id: str, session_id: str = "") -> None:
        """
        Set the current user and session for memory operations.

        Args:
            user_id: Unique identifier for the user.
            session_id: Current session identifier.
        """
        self.current_user_id = user_id
        self.current_session_id = session_id
        logger.info(f"Set current user: {user_id}, session: {session_id}")

    def clear_user_memories(self, user_id: str) -> int:
        """
        Clear all memories for a given user (privacy feature).

        Args:
            user_id: The user whose data should be cleared.

        Returns:
            int: Number of memories deleted.
        """
        deleted = self.memory_manager.clear_user_data(user_id)
        logger.info(f"Cleared {deleted} memories for user '{user_id}'")
        return deleted

    def get_user_memory_count(self, user_id: str) -> int:
        """
        Get the number of memories stored for a user.

        Args:
            user_id: The user to check.

        Returns:
            int: Number of memory entries.
        """
        return self.memory_manager.get_user_memory_count(user_id)

    def _handle_clear_data(self, question: str) -> Dict:
        """
        Handle the "clear my data" command from the user.

        Args:
            question: The user's message (used for response generation).

        Returns:
            Dict: Response indicating data was cleared.
        """
        if not self.current_user_id:
            answer = (
                "I don't have any data stored for you at the moment. "
                "No information to clear!"
            )
        else:
            deleted = self.clear_user_memories(self.current_user_id)
            if deleted > 0:
                answer = (
                    f"✅ I've cleared {deleted} memories I had stored about you. "
                    "Your personal information has been completely removed from my memory database."
                )
            else:
                answer = (
                    "I don't have any data stored for you at the moment. "
                    "No information to clear!"
                )

        self._update_history(question, answer)
        self.last_tool_debug = {"routing": "Privacy (Clear Data)"}

        return {
            "answer": answer,
            "retrieved_chunks": [],
            "confidence": 1.0,
            "response_time": "0.01s",
            "chunk_count": 0,
            "tool_debug": self.last_tool_debug,
            "routing": "Privacy (Clear Data)",
            "memories_used": 0,
            "memories_stored": 0,
        }
