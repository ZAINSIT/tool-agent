import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from duckduckgo_search import DDGS

load_dotenv()

# Initialize client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = "gemini-3.1-flash-lite"# Free-tier friendly [citation:4][citation:9]

# --- Tool 1: Document Retriever (mock) ---
def search_docs(query: str) -> str:
    """Searches a local knowledge base for information about a given query.
    Use this when the user asks about internal documents, policies, or specific stored text.
    """
    # Hardcoded docs for testing the loop. Replace with real retrieval later.
    docs = {
        "razorpay": "Razorpay AI Builder program: Submit a GitHub repo and a 2-min demo video. Shortlisted builders go straight to a panel.",
        "recon": "The Recon Agent is a three-stage settlement reconciliation engine using Python and Gemini. It achieved 86% match rate at 99% precision on a held-out split.",
        "sit": "Srinivas Institute of Technology (SIT) is located in Mangalore, India. The CSBS program is a 4-year B.E. degree."
    }
    query_lower = query.lower()
    results = [text for key, text in docs.items() if key in query_lower]
    if results:
        return " | ".join(results)
    return "No relevant documents found."

# --- Tool 2: Web Search (mock) ---
def web_search(query: str) -> str:
    """Searches the web for current news or external information."""
    if "fail" in query.lower():
        raise Exception("Simulated web search failure.")
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No web results found."
        formatted = "\n".join(
            f"- {r['title']}: {r['body'][:200]}" for r in results
        )
        return formatted
    except Exception as e:
        raise Exception(f"Web search failed: {str(e)}")

# --- Tool registry & schema definitions ---
TOOL_MAP = {
    "search_docs": search_docs,
    "web_search": web_search,
}

# Define function declarations for the model [citation:2]
tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_docs",
            description="Searches local documents for internal knowledge, policies, or stored text.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="The search query for local docs")
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="web_search",
            description="Searches the web for current news or external information.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="The web search query")
                },
                required=["query"],
            ),
        ),
    ]
)

config = types.GenerateContentConfig(tools=[tools])

# --- The Agent Loop ---

def run_agent(user_input: str, max_iterations: int = 5, logger=print):
    logger(f"\n👤 User: {user_input}")
    logger("-" * 50)

    # Initialize conversation history
    history = [
        types.Content(role="user", parts=[types.Part(text=user_input)])
    ]

    for iteration in range(max_iterations):
        logger(f"🤔 Iteration {iteration + 1}: Calling model...")

        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=config,
        )

        # Check if the model wants to call a function
        if response.function_calls:
            function_call = response.function_calls[0]  # Handle first call for simplicity
            name = function_call.name
            args = function_call.args

            logger(f"🔧 Model chose tool: {name}({args})")

            # Execute the tool with error handling
            try:
                if name in TOOL_MAP:
                    args = dict(function_call.args)
                    result = TOOL_MAP[name](**args)
                    result = TOOL_MAP[name](**args)
                    logger(f"📄 Tool result: {result[:150]}...")
                else:
                    result = f"Error: Unknown tool '{name}'"
            except Exception as e:
                # Feed the error back to the model so it can recover [citation:2]
                result = f"Tool error in {name}: {str(e)}"
                logger(f"❌ Tool error in {name}: {str(e)}")

            # Append the model's function call and the tool result to history
            history.append(response.candidates[0].content)
            history.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=name,
                            response={"result": result},
                        )
                    ],
                )
            )
            # Continue loop to let model process the result
            continue

        else:
            # No function call, model produced final answer
            logger(f"✅ Final answer: {response.text}")
            logger("-" * 50)
            return response.text

    logger("⚠️ Max iterations reached without final answer.")
    return None


# --- Test Scenarios ---
if __name__ == "__main__":
    print("=" * 50)
    print("TEST 1: Single tool call (local docs)")
    run_agent("What does my document say about the Razorpay AI Builder?")

    print("\n" + "=" * 50)
    print("TEST 2: Multi-step (docs then web)")
    run_agent("Search my docs for the recon agent, then search the web for recent AI news.")

    print("\n" + "=" * 50)
    print("TEST 3: Tool failure recovery")
    run_agent("Search the web for 'fail' to see if it handles errors gracefully.")