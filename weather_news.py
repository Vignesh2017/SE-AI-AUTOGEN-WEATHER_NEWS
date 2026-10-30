from dotenv import load_dotenv
import asyncio
import logging
import os
import re

import requests
from ddgs import DDGS

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()
# ============================================================
# 1. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("weather_news")


# ============================================================
# 2. WEATHER INFORMATION VALIDATION
# ============================================================

WEATHER_TERMS = [
    "weather", "forecast", "temperature", "humidity",
    "rainfall", "precipitation", "wind speed", "wind",
    "sunny", "cloudy", "overcast", "rain", "rainy",
    "showers", "thunderstorm", "storm", "heatwave",
    "degrees", "monsoon", "drizzle", "snow",
    "feels like", "weather alert"
]

WEATHER_DETAILS = [
    "partly cloudy", "mostly cloudy", "clear skies",
    "heavy rain", "light rain", "thunderstorm",
    "high of", "low of", "humidity",
    "wind speed", "chance of rain", "rainfall",
    "feels like", "weather alert", "forecast",
    "sunny", "overcast", "showers", "degrees"
]


def contains_weather_information(results: list) -> bool:
    """
    Check whether search results contain meaningful weather
    information rather than unrelated general news.
    """

    if not results:
        logger.warning("[VALIDATION] No search results to validate.")
        return False

    for item in results:

        content = (
            item.get("title", "") + " " +
            item.get("body", "")
        ).lower()

        has_weather_context = any(
            term in content for term in WEATHER_TERMS
        )

        has_weather_detail = (
            bool(re.search(r"\b\d{1,2}\s*°\s*[cf]\b", content))
            or bool(re.search(r"\b\d{1,2}\s*degrees\b", content))
            or any(
                detail in content
                for detail in WEATHER_DETAILS
            )
        )

        if has_weather_context and has_weather_detail:
            logger.info(
                "[VALIDATION] Weather information detected."
            )
            return True

    logger.warning(
        "[VALIDATION] Results do not contain sufficient "
        "weather information."
    )

    return False


# ============================================================
# 3. DUCKDUCKGO SEARCH - PRIMARY PROVIDER
# ============================================================

def search_with_duckduckgo(query: str) -> list:
    """
    Search DuckDuckGo for current weather conditions,
    temperature, forecast and weather-related news.
    """

    logger.info("[SEARCH] Invoking DuckDuckGo")
    logger.info("[SEARCH] Query: %s", query)

    results = DDGS().text(
        query,
        max_results=5
    )

    formatted_results = [
        {
            "title": item.get("title", ""),
            "body": item.get("body", ""),
            "url": item.get("href", "")
        }
        for item in results
    ]

    logger.info(
        "[SEARCH] DuckDuckGo returned %d results",
        len(formatted_results)
    )

    return formatted_results


# ============================================================
# 4. SERPER GOOGLE SEARCH - FALLBACK PROVIDER
# ============================================================

def search_with_serper(query: str) -> list:
    """
    Search Google using the Serper API for current weather
    conditions, forecast, temperature and weather news.
    """

    logger.info("[SEARCH] Invoking Google Search via Serper")
    logger.info("[SEARCH] Query: %s", query)

    api_key = os.getenv("SERPER_API_KEY")

    if not api_key:
        raise ValueError(
            "SERPER_API_KEY environment variable is missing."
        )

    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        },
        json={
            "q": query,
            "num": 5
        },
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    formatted_results = [
        {
            "title": item.get("title", ""),
            "body": item.get("snippet", ""),
            "url": item.get("link", "")
        }
        for item in data.get("organic", [])
    ]

    logger.info(
        "[SEARCH] Serper returned %d results",
        len(formatted_results)
    )

    return formatted_results


# ============================================================
# 5. WEATHER NEWS SEARCH TOOL
# ============================================================

def search_weather_news(city: str) -> str:
    """
    Search for current weather conditions, temperature, humidity,
    forecasts and weather-related news for a city.

    First use DuckDuckGo. If DuckDuckGo fails, returns no results,
    or returns results without meaningful weather information,
    automatically invoke Google Search through Serper.

    Return validated search results with their source URLs.
    """

    query = (
        f"{city} current weather temperature humidity "
        f"forecast today rainfall weather news"
    )

    results = []
    provider = None

    # --------------------------------------------------------
    # STEP 1: PRIMARY SEARCH - DUCKDUCKGO
    # --------------------------------------------------------

    try:

        logger.info("[PRIMARY] Starting DuckDuckGo search")

        results = search_with_duckduckgo(query)

        if contains_weather_information(results):

            provider = "DuckDuckGo"

            logger.info(
                "[SUCCESS] DuckDuckGo provided weather information."
            )

        else:

            logger.warning(
                "[FALLBACK] DuckDuckGo returned irrelevant "
                "or insufficient weather information."
            )

    except Exception:

        logger.exception(
            "[FALLBACK] DuckDuckGo search failed."
        )

    # --------------------------------------------------------
    # STEP 2: FALLBACK SEARCH - SERPER GOOGLE
    # --------------------------------------------------------

    if provider is None:

        try:

            logger.info(
                "[FALLBACK] Starting Google Search through Serper"
            )

            fallback_query = (
                f"{city} current weather conditions temperature "
                f"humidity forecast today weather report"
            )

            results = search_with_serper(fallback_query)

            if contains_weather_information(results):

                provider = "Google Search via Serper"

                logger.info(
                    "[SUCCESS] Serper provided weather information."
                )

            else:

                logger.warning(
                    "[VALIDATION] Serper results also lack "
                    "sufficient weather information."
                )

        except Exception:

            logger.exception(
                "[SEARCH] Serper fallback failed."
            )

    # --------------------------------------------------------
    # STEP 3: HANDLE FAILURE
    # --------------------------------------------------------

    if provider is None:

        logger.error(
            "[SEARCH] Both providers failed to provide "
            "validated weather information for %s",
            city
        )

        return (
            f"Unable to retrieve verified weather information "
            f"for {city}. DuckDuckGo and Serper did not provide "
            f"sufficient weather results."
        )

    # --------------------------------------------------------
    # STEP 4: FORMAT SEARCH RESULTS
    # --------------------------------------------------------

    logger.info(
        "[SEARCH] Final selected provider: %s",
        provider
    )

    formatted_results = []

    for index, item in enumerate(results, start=1):

        formatted_results.append(
            f"Result {index}\n"
            f"Title: {item['title']}\n"
            f"Description: {item['body']}\n"
            f"Source: {item['url']}"
        )

    return (
        f"Selected Provider: {provider}\n\n"
        + "\n\n".join(formatted_results)
    )


# ============================================================
# 6. AUTOGEN WEATHER NEWS AGENT
# ============================================================

async def main() -> None:

    # Get city from user
    city = input(
        "Enter a city to check its weather news: "
    ).strip()

    if not city:
        logger.error("City cannot be empty.")
        return

    # Read OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is missing."
        )

    # Initialize OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=openai_key
    )

    try:

        # Create AutoGen AssistantAgent
        agent = AssistantAgent(
            name="weather_news_agent",

            model_client=model_client,

            tools=[
                search_weather_news
            ],

            reflect_on_tool_use=True,

            model_client_stream=True,

            system_message=(
                "You are a Weather News Assistant. "

                "Use the search_weather_news tool to retrieve "
                "weather information for the requested city. "

                "The tool uses DuckDuckGo as its primary search "
                "provider and Google Search through Serper as "
                "a fallback when weather information is missing. "

                "Use only information supported by the search results. "

                "Do not invent temperatures, forecasts, humidity "
                "or weather alerts. "

                "Do not treat search-result snippets as guaranteed "
                "live weather measurements. "

                "If the information is unavailable or insufficient, "
                "clearly communicate that limitation. "

                "Return exactly 3 short lines with no blank lines. "

                "Line 1: Current weather conditions or temperature "
                "if available. "

                "Line 2: Forecast or relevant weather news. "

                "Line 3: Source URL or information limitation. "

                "Do not add headings, introductions or conclusions."
            )
        )

        # Prepare agent task
        task = (
            f"Find the latest available weather conditions, "
            f"temperature, forecast and relevant weather news "
            f"for {city}. "

            "You must use the search_weather_news tool. "
            "Return exactly 3 short lines."
        )

        logger.info(
            "[AGENT] Starting Weather News Agent for %s",
            city
        )

        logger.info("[AGENT] Streaming enabled.")

        print(f"\nWeather News for {city}\n")

        # Stream agent events and model response
        await Console(
            agent.run_stream(task=task),
            output_stats=True
        )

        logger.info("[AGENT] Execution completed.")

    except Exception:

        logger.exception("[AGENT] Execution failed.")
        raise

    finally:

        # Close model client
        await model_client.close()

        logger.info(
            "[CLEANUP] OpenAI model client closed."
        )


# ============================================================
# 7. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())
