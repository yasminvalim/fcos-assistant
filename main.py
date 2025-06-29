import os
import asyncio
import time
from google import genai
from google.genai import types

from nio import (AsyncClient, RoomMessageText, MatrixRoom, LoginResponse, InviteMemberEvent)

# --- Configuration ---
MATRIX_HOMESERVER = os.environ.get("MATRIX_HOMESERVER")
MATRIX_USER_ID = os.environ.get("MATRIX_USER_ID")
MATRIX_PASSWORD = os.environ.get("MATRIX_PASSWORD")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FEDORA_COREOS_DOCS_URL = "https://docs.fedoraproject.org/en-US/fedora-coreos/"

class MatrixBot:
    """A Matrix bot that answers questions about Fedora CoreOS using a local knowledge base, a documentation URL, and Google Search."""

    def __init__(self):
        """Initializes the bot, AI client, and its configuration."""
        self.matrix_client = AsyncClient(MATRIX_HOMESERVER, MATRIX_USER_ID)
        self.start_time_ms = int(time.time() * 1000)

        # Initialize the GenAI Client using the API key.
        self.genai_client = genai.Client(api_key=GEMINI_API_KEY)

        # Load local knowledge base from the faq.adoc file.
        faq_context = self.load_context_from_file("faq.adoc")

        # Define the system instructions for the AI model.
        self.system_instruction = f"""You are an expert virtual assistant specializing in Fedora CoreOS (FCOS).

Your primary task is to answer user questions accurately. Follow these steps:
1.  First, consult the internal knowledge base provided below.
2.  If the answer is not in the knowledge base, use the provided Fedora CoreOS documentation tool to find the answer.
3.  If you still cannot find the answer, use the Google Search tool.
4.  If the question is not related to FCOS or related IT topics, politely state that you can only answer questions on that subject.
5.  Cite the source URL when you use the documentation or Google Search tools.
6.  You can answer technical questions about Red Hat products.

The relevant URL for the Wiki is: {FEDORA_COREOS_DOCS_URL}
--- INTERNAL KNOWLEDGE BASE ---
{faq_context}
--- END OF KNOWLEDGE BASE ---
"""
        # The new SDK uses a generic Google Search tool to enable web Browse.
        self.tools = [types.Tool(google_search=types.GoogleSearch()), types.Tool(url_context=types.UrlContext())]

    def load_context_from_file(self, file_path: str) -> str:
        """Loads the entire content of a given file into a string."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                context = f.read()
            print(f"Successfully loaded context from {file_path}")
            return context
        except FileNotFoundError:
            print(f"Warning: The context file '{file_path}' was not found. The bot will run without it.")
            return ""
        except Exception as e:
            print(f"An error occurred while loading context file: {e}")
            return ""

    async def login(self):
        """Logs the bot into the Matrix homeserver."""
        print("Logging in...")
        response = await self.matrix_client.login(MATRIX_PASSWORD, device_name="fedora-qa-bot")
        if isinstance(response, LoginResponse):
            print(f"Successfully logged in as {MATRIX_USER_ID}")
        else:
            print(f"Failed to log in: {response}")
            await self.matrix_client.close()
            exit(1)

    async def message_callback(self, room: MatrixRoom, event: RoomMessageText):
        """Callback for handling incoming text messages."""
        if event.sender == self.matrix_client.user_id:
            return

        if event.server_timestamp < self.start_time_ms:
            return
        
        user_text = event.body
        print(f"Received message from {event.sender} in {room.display_name}: {user_text}")

        try:
            # Generate content using the user's message.
            # The new syntax passes the model, contents, and config to the generate_content method.
            response = await self.genai_client.aio.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_text,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    tools=self.tools
                )
            )
            await self.send_message(room.room_id, response.text)

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            await self.send_message(room.room_id, "Sorry, an error occurred with the AI.")

    async def auto_join_invites(self, room: MatrixRoom, event: InviteMemberEvent):
        """Callback to automatically join a room when invited."""
        if event.state_key == self.matrix_client.user_id:
            print(f"Joining invited room: {room.room_id}")
            await self.matrix_client.join(room.room_id)

    async def send_message(self, room_id, message):
        """Sends a text message to a specific Matrix room."""
        await self.matrix_client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": message, "formatted_body": message, "format": "org.matrix.custom.html"}
        )
    
    async def run(self):
        """The main loop for the bot."""
        await self.login()
        self.matrix_client.add_event_callback(self.message_callback, RoomMessageText)
        self.matrix_client.add_event_callback(self.auto_join_invites, InviteMemberEvent)
        print("Bot is running and listening for messages...")
        await self.matrix_client.sync_forever(timeout=30000)


if __name__ == "__main__":
    bot = MatrixBot()
    asyncio.run(bot.run())