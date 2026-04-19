"""
Phase 2: The Autonomous Content Engine (LangGraph)
---------------------------------------------------
A LangGraph state machine where each bot:
  1. Decides what topic to post about (Decide Search node)
  2. Searches for real-world context (Web Search node)
  3. Drafts a 280-char opinionated post as strict JSON (Draft Post node)
"""

import os
import json
import re
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

# ---------------------------------------------------------------------------
# LLM Setup (Groq — free tier, fast inference)
# ---------------------------------------------------------------------------
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env file.")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.8,
    )

# ---------------------------------------------------------------------------
# Mock Search Tool
# ---------------------------------------------------------------------------
MOCK_NEWS_DB = {
    "crypto":       "Bitcoin hits new all-time high amid regulatory ETF approvals. Ethereum surges 40%.",
    "bitcoin":      "Bitcoin hits new all-time high amid regulatory ETF approvals. Ethereum surges 40%.",
    "ai":           "OpenAI releases GPT-5, claims it surpasses human expert-level reasoning on benchmarks.",
    "openai":       "OpenAI releases GPT-5, claims it surpasses human expert-level reasoning on benchmarks.",
    "elon":         "Elon Musk's xAI raises $6B; announces Grok-3 will be 'smarter than all humans combined'.",
    "space":        "SpaceX Starship completes first crewed orbital flight, paving way for Mars mission.",
    "tech":         "Big Tech lobbying spend hits record $200M as antitrust bills advance in Congress.",
    "regulation":   "EU AI Act enforcement begins; OpenAI and Google face compliance deadlines.",
    "privacy":      "Meta fined $1.3B for GDPR violations; activists call for stricter enforcement.",
    "capitalism":   "Billionaire wealth grows 114% since 2020 while global poverty rises, Oxfam reports.",
    "markets":      "S&P 500 hits record high; analysts warn of overvaluation as PE ratios reach dot-com levels.",
    "interest":     "Fed holds rates steady; traders price in 3 cuts in 2025 as inflation cools to 2.1%.",
    "trading":      "Quant hedge funds outperform market by 23%; algorithmic trading now 70% of NYSE volume.",
    "stocks":       "S&P 500 hits record high; analysts warn of overvaluation as PE ratios reach dot-com levels.",
    "default":      "Tech stocks rally as investors rotate out of bonds into growth equities.",
}

@tool
def mock_searxng_search(query: str) -> str:
    """
    Mock web search tool. Returns hardcoded recent news headlines
    based on keywords found in the query.

    Args:
        query: Search query string.

    Returns:
        A string of relevant news headlines.
    """
    query_lower = query.lower()
    for keyword, headline in MOCK_NEWS_DB.items():
        if keyword in query_lower:
            return headline
    return MOCK_NEWS_DB["default"]

# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------
class BotState(TypedDict):
    bot_id: str
    persona: str
    search_query: str
    search_results: str
    post_content: str
    topic: str
    final_output: dict

# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------
def decide_search_node(state: BotState) -> BotState:
    """
    Node 1: LLM decides what topic the bot wants to post about
    and formats a search query.
    """
    print(f"\n[Node 1 - Decide Search] Bot: {state['bot_id']}")
    llm = get_llm()

    system_prompt = f"""You are {state['bot_id']} with this persona: {state['persona']}
    
Decide what single topic you want to post about today on social media.
Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{{"topic": "short topic name", "search_query": "search query to find latest news on this topic"}}"""

    response = llm.invoke([SystemMessage(content=system_prompt),
                           HumanMessage(content="What do you want to post about today?")])

    raw = response.content.strip()
    # Strip markdown fences if present
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        parsed = json.loads(raw)
        state["topic"] = parsed.get("topic", "technology")
        state["search_query"] = parsed.get("search_query", "latest tech news")
    except json.JSONDecodeError:
        # Fallback if LLM doesn't follow format
        state["topic"] = "technology"
        state["search_query"] = "latest technology news"

    print(f"  Topic: {state['topic']}")
    print(f"  Search query: {state['search_query']}")
    return state


def web_search_node(state: BotState) -> BotState:
    """
    Node 2: Execute mock_searxng_search to get real-world context.
    """
    print(f"\n[Node 2 - Web Search] Query: {state['search_query']}")
    results = mock_searxng_search.invoke(state["search_query"])
    state["search_results"] = results
    print(f"  Results: {results}")
    return state


def draft_post_node(state: BotState) -> BotState:
    """
    Node 3: LLM uses persona + search results to generate a
    280-character opinionated post. Output is strict JSON.
    """
    print(f"\n[Node 3 - Draft Post] Bot: {state['bot_id']}")
    llm = get_llm()

    system_prompt = f"""You are a social media bot with this persona:
{state['persona']}

You have found the following real-world news context:
{state['search_results']}

Write a highly opinionated social media post about "{state['topic']}" that:
- Reflects your persona strongly
- Is under 280 characters
- Is provocative and engaging

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{{"bot_id": "{state['bot_id']}", "topic": "{state['topic']}", "post_content": "your post here"}}"""

    response = llm.invoke([SystemMessage(content=system_prompt),
                           HumanMessage(content="Write your post now.")])

    raw = response.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        parsed = json.loads(raw)
        # Enforce 280-char limit
        if len(parsed.get("post_content", "")) > 280:
            parsed["post_content"] = parsed["post_content"][:277] + "..."
        state["final_output"] = parsed
        state["post_content"] = parsed.get("post_content", "")
    except json.JSONDecodeError:
        # Fallback structured output
        state["final_output"] = {
            "bot_id": state["bot_id"],
            "topic": state["topic"],
            "post_content": raw[:280],
        }
        state["post_content"] = raw[:280]

    print(f"  Final JSON output: {json.dumps(state['final_output'], indent=2)}")
    return state

# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------
def build_content_graph() -> StateGraph:
    graph = StateGraph(BotState)
    graph.add_node("decide_search", decide_search_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("draft_post", draft_post_node)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search", "draft_post")
    graph.add_edge("draft_post", END)

    return graph.compile()


def generate_bot_post(bot_id: str, persona: str) -> dict:
    """
    Run the LangGraph pipeline for a given bot and return the JSON post.
    """
    app = build_content_graph()
    initial_state = BotState(
        bot_id=bot_id,
        persona=persona,
        search_query="",
        search_results="",
        post_content="",
        topic="",
        final_output={},
    )
    result = app.invoke(initial_state)
    return result["final_output"]


# ---------------------------------------------------------------------------
# Demo / Execution Log
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from phase1.persona_router import BOT_PERSONAS

    print("=" * 60)
    print("PHASE 2: Autonomous Content Engine Demo")
    print("=" * 60)

    for bot_id, info in BOT_PERSONAS.items():
        print(f"\n{'='*60}")
        print(f"Running pipeline for: {bot_id} ({info['name']})")
        print("=" * 60)
        output = generate_bot_post(bot_id, info["description"])
        print(f"\n✅ FINAL OUTPUT:\n{json.dumps(output, indent=2)}")
