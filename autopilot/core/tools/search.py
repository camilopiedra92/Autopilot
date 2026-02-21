"""
Web Search Tool — V3 Phase 7+: Platform-level tool for autonomous web research.

Provides real-time web search capabilities to agents using DuckDuckGo.
"""

import json
import structlog
from ddgs import DDGS

from autopilot.core.tools.registry import tool

logger = structlog.get_logger(__name__)


@tool(name="search_web", description="Searches the web for information using a search engine.")
def search_web(query: str, max_results: int = 3, region: str = "wt-wt") -> str:
    """
    Perform a real web search to find current information, documentation,
    or entity resolution (e.g., finding the real name of a merchant).

    Args:
        query: The search query to run.
        max_results: Maximum number of results to return (default: 3, max: 10).
        region: Geographic region for search results (default: "wt-wt" for worldwide).
                Use "es-co" for Colombia, "en-us" for US, etc.

    Returns:
        JSON string containing the search results (title, snippet, url).
    """
    logger.info("searching_web", query=query, max_results=max_results, region=region)
    try:
        max_results = min(max(1, max_results), 10) # Clamp between 1 and 10
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results, region=region)
            
            formatted_results = []
            if results:
                for r in results:
                    formatted_results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", "")
                    })
            
            return json.dumps(formatted_results, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error("web_search_failed", error=str(e), query=query)
        return json.dumps({"error": f"Search failed: {str(e)}"})
