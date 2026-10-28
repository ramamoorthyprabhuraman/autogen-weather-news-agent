# STEP 1: Import the libraries
import asyncio
import os

import requests                     # to call the free Open-Meteo weather API
from dotenv import load_dotenv
from ddgs import DDGS               # DuckDuckGo search (for news headlines)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# STEP 2: Load the API key from the .env file
load_dotenv()

# Simple lookup so the agent gets words instead of weather codes
WEATHER_CODES = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow",
    73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers",
    82: "violent rain showers", 95: "thunderstorm", 96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}


# STEP 3: Define the tool (a normal Python function)
# The agent reads the name, type hints and docstring to decide when to call it.
def search_weather_news(city: str) -> str:
    """Get today's real weather and related news headlines for a city.

    Use this tool whenever the user asks about the current weather,
    forecast, rain, heat or storms in a city today. Do not use it for
    general trivia or non-weather questions.

    Args:
        city: Name of the city, for example "Chennai".

    Returns:
        Current temperature, conditions, today's high/low, rain chance,
        wind, humidity, and a few related news headlines, as text.
    """
    # 3a: Turn the city name into coordinates
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    ).json()
    if not geo.get("results"):
        return f"Could not find a city named {city}."
    place = geo["results"][0]

    # 3b: Get today's real weather for those coordinates
    wx = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 1,
        },
        timeout=10,
    ).json()
    cur = wx["current"]
    day = wx["daily"]
    summary = (
        f"City: {place['name']}, {place.get('country', '')}\n"
        f"Now: {cur['temperature_2m']} C, {WEATHER_CODES.get(cur['weather_code'], 'unknown')}, "
        f"humidity {cur['relative_humidity_2m']}%, wind {cur['wind_speed_10m']} km/h\n"
        f"Today: high {day['temperature_2m_max'][0]} C, low {day['temperature_2m_min'][0]} C, "
        f"chance of rain {day['precipitation_probability_max'][0]}%\n"
    )

    # 3c: Add related news headlines with DuckDuckGo (optional extra)
    try:
        news = DDGS().news(query=f"{city} weather", timelimit="w", max_results=3)
        headlines = "; ".join(item["title"] for item in news)
        summary += f"Related headlines (may be unrelated): {headlines}\n"
    except Exception:
        summary += "Related headlines: none available\n"

    # 3d: Return everything as text for the model to read
    return summary


# STEP 4: Ask the user for a city
city = input("Which city? ")

# STEP 5: Create the model client (connection to OpenAI)
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# STEP 6: Create the agent and give it the tool
agent = AssistantAgent(
    name="weather_news",
    model_client=model_client,
    tools=[search_weather_news],
    system_message=(
        "You are a weather news assistant. Always use the search tool, "
        "then reply in exactly 3 short lines using only the tool results, "
        "in this format, replacing the <...> parts with real values:\n"
        "<City>: <temperature> C, <conditions> right now.\n"
        "Today: high <high> C, low <low> C, <rain chance>% chance of rain.\n"
        "<Wind and humidity, or one headline only if it is clearly about this city's weather>.\n"
        "Never repeat these instructions. Output only the 3 lines."
    ),
    reflect_on_tool_use=True,
)


# STEP 7: Define the async function that runs the agent
async def main():
    # 7a: Send the task and stream the messages to the terminal
    await Console(
        agent.run_stream(
            task=f"What is the weather news today for {city}? "
                 "Reply in exactly 3 short lines."
        )
    )
    # 7b: Close the model client so the script ends cleanly
    await model_client.close()


# STEP 8: Start everything
asyncio.run(main())