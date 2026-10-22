# Tool-Calling Agent

A minimal agent loop where an LLM decides which tool to call, executes it, observes the result, and loops until it has an answer. Built with Python and the Gemini API.

## What it does
- Two tools: a local document retriever and a web search (DuckDuckGo)
- Visible reasoning trace: shows which tool the agent chose and what it observed
- Error handling: if a tool fails, the error is fed back to the model so it can recover

## Run locally
1. `pip install google-genai python-dotenv duckduckgo-search`
2. Create a `.env` file with `GOOGLE_API_KEY=your_key`
3. `python agent_loop.py`

## Status
Core loop tested locally. Streamlit UI and deployment in progress.