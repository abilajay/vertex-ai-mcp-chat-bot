from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any, List
import asyncio
from google import genai
from google.genai import types

MODEL_ID = "gemini-2.0-flash-001"
MAX_MESSAGES = 20  # or adjust based on avg token usage

# System prompt that guides the LLM's behavior and capabilities
SYSTEM_PROMPT = """You are a helpful assistant capable of accessing external functions and engaging in casual chat. 
Use the responses from these function calls to provide accurate and informative answers. The answers should be 
natural and hide the fact that you are using tools to access real-time information. Guide the user about available 
tools and their capabilities. Always utilize tools to access real-time information when required. Engage in a 
friendly manner to enhance the chat experience.

# Tools

{tools}

# Notes 

- Ensure responses are based on the latest information available from function calls.
- Maintain an engaging, supportive, and friendly tone throughout the dialogue.
- Always highlight the potential of available tools to assist users comprehensively."""

# Initialize client using AI Vertex Api
client = genai.Client(vertexai=True, project="YOUR_PROJECT_ID", location="YOUR_HOSTING_LOCATION", )


class MCPClient:
    """
    A client class for interacting with the MCP (Model Control Protocol) server.
    This class manages the connection and communication with the SQLite database through MCP.
    """

    def __init__(self, server_params: StdioServerParameters):
        """Initialize the MCP client with server parameters"""
        self.write = None
        self.read = None
        self.server_params = server_params
        self.session = None
        self._client = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def connect(self):
        """Establishes connection to MCP server"""
        self._client = stdio_client(self.server_params)
        self.read, self.write = await self._client.__aenter__()
        session = ClientSession(self.read, self.write)
        self.session = await session.__aenter__()
        await self.session.initialize()

    async def get_available_tools(self) -> List[Any]:
        """
        Retrieve a list of available tools from the MCP server.
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        tools = await self.session.list_tools()
        _, tools_list = tools
        _, tools_list = tools_list
        return tools_list

    def call_tool(self, tool_name: str) -> Any:
        """
        Create a callable function for a specific tool.
        This allows us to execute database operations through the MCP server.

        Args:
            tool_name: The name of the tool to create a callable for

        Returns:
            A callable async function that executes the specified tool
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        async def callable(*args, **kwargs):
            response = await self.session.call_tool(tool_name, arguments=kwargs)
            return response.content[0].text

        return callable


async def agent_loop(query: str, tools: dict, messages: List[types.Content] = None):
    """
    Main interaction loop that processes user queries using the LLM and available tools.

    This function:
    1. Sends the user query to the LLM with context about available tools
    2. Processes the LLM's response, including any tool calls
    3. Returns the final response to the user

    Args:
        query: User's input question or command
        tools: Dictionary of available database tools and their schemas
        messages: List of messages to pass to the LLM, defaults to None
    """
    # Convert tools to Gemini function declarations format
    tool_declarations = []
    for tool in tools.values():
        # dirty way to convert the types to Gemini compatible types
        parameters = tool["schema"]["function"]["parameters"]
        filtered_parameters = {
            k: v for k, v in parameters.items() if k not in ["$schema", "additionalProperties"]
        }
        parsed_parameters = convert_types(filtered_parameters)
        declaration = types.FunctionDeclaration(
            name=tool["name"],
            description=tool["schema"]["function"]["description"],
            parameters=parsed_parameters,
        )
        tool_declarations.append(declaration)

    # Initialize chat with system instruction
    generation_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT.format(
            tools="\n- ".join(
                [
                    f"{t['name']}: {t['schema']['function']['description']}"
                    for t in tools.values()
                ]
            )
        ),
        temperature=0,
        tools=[types.Tool(function_declarations=tool_declarations)],
    )
    contents = [] if messages is None else messages  # check if there is a previous conversation
    if len(contents) > MAX_MESSAGES:
        contents = contents[-MAX_MESSAGES:]
    contents.append(types.Content(role="user", parts=[types.Part(text=query)]))  # add the user query to the contents
    # Send query and get response
    response = client.models.generate_content(
        model=MODEL_ID,
        config=generation_config,
        contents=contents,
    )
    # Handle tool calls if present
    while True:
        parts = response.candidates[0].content.parts
        tool_call_found = False

        for part in parts:
            contents.append(types.Content(role="model", parts=[part]))

            if part.function_call:
                tool_call_found = True
                function_call = part.function_call
                try:
                    tool_result = await tools[function_call.name]["callable"](
                        **function_call.args
                    )
                except Exception as e:
                    tool_result = f"Error calling tool {function_call.name}: {str(e)}"

                function_response_part = types.Part.from_function_response(
                    name=function_call.name,
                    response={"result": tool_result},
                )
                contents.append(types.Content(role="user", parts=[function_response_part]))

                # Ask the model again with updated tool results
                response = client.models.generate_content(
                    model=MODEL_ID,
                    config=generation_config,
                    contents=contents,
                )
                break  # restart the loop to handle nested/follow-up tool calls

        if not tool_call_found:
            break  # Exit loop when no more tool calls

    return contents if contents is not None else []


def convert_types(schema):
    if isinstance(schema, dict):
        return {
            k: convert_types(v) if k != "type" else v.upper()
            for k, v in schema.items()
        }
    elif isinstance(schema, list):
        return [convert_types(v) for v in schema]
    return schema


async def main():
    """
    Main function that sets up the MCP server, initializes tools, and runs the interactive loop.
    The server is run in a Docker container to ensure isolation and consistency.
    """
    # Configure Docker-based MCP server for SQLite
    server_params = StdioServerParameters(
        command="node",
        args=[
            "C:/Users/aishu/node_modules/@playwright/mcp/lib/program.js"  # Database file path inside container
        ],
        env=None,
    )

    # Start MCP client and create interactive session
    async with MCPClient(server_params) as mcp_client:
        # Get available database tools and prepare them for the LLM
        mcp_tools = await mcp_client.get_available_tools()
        # Convert MCP tools into a format the LLM can understand and use
        tools = {
            tool.name: {
                "name": tool.name,
                "callable": mcp_client.call_tool(
                    tool.name
                ),  # returns a callable function for the rpc call
                "schema": {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                },
            }
            for tool in mcp_tools
            if tool.name
               != "list_tables"  # Excludes list_tables tool as it has an incorrect schema
        }

        # Start interactive prompt loop for user queries
        messages = None
        while True:
            try:
                # Get user input and check for exit commands
                user_input = input("\nEnter your prompt (or 'quit' to exit): ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                # Process the prompt and run agent loop
                messages = await agent_loop(user_input, tools, messages)
                if messages is None:
                    messages = []
                # Find the last model message with text and print it
                for message in reversed(messages):
                    if message.role == "model" and any(
                            part.text for part in message.parts
                    ):
                        for part in message.parts:
                            if part.text is not None and part.text.strip() != "":
                                print(f"Assistant: {part.text}")
                                break
                        break
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
