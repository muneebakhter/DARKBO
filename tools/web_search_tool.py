"""
Web Search Tool for DARKBO
Provides web search capabilities using DuckDuckGo search
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
import requests
from urllib.parse import quote_plus

from .base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    """Tool for performing web searches"""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for current information using DuckDuckGo",
            version="1.0.0"
        )
        self.base_url = "https://api.duckduckgo.com/"
    
    async def execute(self, 
                     query: str,
                     max_results: Optional[int] = 5,
                     safe_search: Optional[str] = "moderate",
                     **kwargs) -> ToolResult:
        """
        Execute the web search tool
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 5)
            safe_search: Safe search setting - "strict", "moderate", or "off" (default: "moderate")
            **kwargs: Additional parameters (ignored)
        
        Returns:
            ToolResult with search results
        """
        start_time = time.time()
        
        if not query or not query.strip():
            return ToolResult(
                success=False,
                error="Search query cannot be empty",
                execution_time=time.time() - start_time
            )
        
        try:
            # Use DuckDuckGo Instant Answer API
            # This is a simple implementation - in production you might want to use a proper search API
            search_results = await self._search_duckduckgo(query, max_results, safe_search)
            
            result_data = {
                "query": query,
                "results": search_results,
                "total_results": len(search_results),
                "max_results_requested": max_results,
                "safe_search": safe_search,
                "search_engine": "DuckDuckGo"
            }
            
            return ToolResult(
                success=True,
                data=result_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Web search error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    async def _search_duckduckgo(self, query: str, max_results: int, safe_search: str) -> List[Dict[str, Any]]:
        """
        Perform search using DuckDuckGo Instant Answer API
        
        Note: This is a simplified implementation. For production use, consider:
        - Using official search APIs (Google Custom Search, Bing Search API)
        - Implementing proper rate limiting
        - Adding caching mechanisms
        - Better error handling and retries
        """
        
        try:
            # DuckDuckGo Instant Answer API
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            # Make the request in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.get(self.base_url, params=params, timeout=10)
            )
            
            if response.status_code != 200:
                return [{
                    "title": "Search Error",
                    "snippet": f"Search service returned status code {response.status_code}",
                    "url": "",
                    "source": "error"
                }]
            
            data = response.json()
            results = []
            
            # Extract instant answer if available
            if data.get('Abstract'):
                results.append({
                    "title": data.get('Heading', 'Instant Answer'),
                    "snippet": data.get('Abstract'),
                    "url": data.get('AbstractURL', ''),
                    "source": data.get('AbstractSource', 'DuckDuckGo')
                })
            
            # Extract related topics
            for topic in data.get('RelatedTopics', []):
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        "title": topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else topic.get('Text', ''),
                        "snippet": topic.get('Text', ''),
                        "url": topic.get('FirstURL', ''),
                        "source": "DuckDuckGo Related"
                    })
                
                if len(results) >= max_results:
                    break
            
            # If we don't have enough results, add a fallback search suggestion
            if len(results) == 0:
                results.append({
                    "title": f"Search for: {query}",
                    "snippet": f"No instant answers found. Try searching for '{query}' on a search engine for more detailed results.",
                    "url": f"https://duckduckgo.com/?q={quote_plus(query)}",
                    "source": "DuckDuckGo Search Link"
                })
            
            return results[:max_results]
            
        except requests.exceptions.RequestException as e:
            # Return a helpful error message as a search result
            return [{
                "title": "Search Service Unavailable", 
                "snippet": f"Web search is currently unavailable due to network restrictions. For information about '{query}', please check your local knowledge base or contact support directly.",
                "url": "",
                "source": "error"
            }]
        except Exception as e:
            return [{
                "title": "Search Error",
                "snippet": f"An error occurred while searching: {str(e)}",
                "url": "",
                "source": "error"
            }]
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for web search tool parameters"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                    "minLength": 1,
                    "examples": ["current weather in New York", "latest Python version", "how to install Docker"]
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                },
                "safe_search": {
                    "type": "string",
                    "description": "Safe search setting",
                    "enum": ["strict", "moderate", "off"],
                    "default": "moderate"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }