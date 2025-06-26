import os
import asyncio
import google.generativeai as genai
from nio import (AsyncClient, RoomMessageText, MatrixRoom, LoginResponse, InviteMemberEvent)

MATRIX_HOMESERVER = os.environ.get("MATRIX_HOMESERVER")
MATRIX_USER_ID = os.environ.get("MATRIX_USER_ID")
MATRIX_PASSWORD = os.environ.get("MATRIX_PASSWORD")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)


FEDORA_COREOS_QA = {
    "what is": "Fedora CoreOS is an automatically updating, minimal, monolithic, container-focused operating system, designed for clusters but also operable standalone. It is the community-driven upstream for RHEL CoreOS.",
    "update": "Updates are managed by `rpm-ostree` and are delivered automatically. The `zincati` client is the agent that manages these automatic updates, applying them transactionally.",
    "ignition": "Ignition is the utility used to provision Fedora CoreOS on the first boot. It configures disks, formatting, users, and files using a JSON configuration.",
    "butane": "Butane is a tool that helps create Ignition configuration files. You write in a more user-friendly YAML format, and Butane converts it to the JSON that Ignition understands. It was formerly known as the Fedora CoreOS Config Transpiler (FCCT).",
    "rpm-ostree": "`rpm-ostree` is the hybrid image/package system that Fedora CoreOS is built on. It enables atomic upgrades, rollbacks, and allows you to layer traditional RPM packages on top of the base OS image if needed.",
    "relation to rhel": "Fedora CoreOS is the open-community upstream project where new features are developed and tested. RHEL CoreOS is the downstream, enterprise-grade product with long-term commercial support from Red Hat."
}

class MatrixBot:
    def __init__(self):
        self.client = AsyncClient(MATRIX_HOMESERVER, MATRIX_USER_ID)
        self.ai_model = genai.GenerativeModel('gemini-1.5-flash-latest')

    async def login(self):
        print("Logging in...")
        response = await self.client.login(MATRIX_PASSWORD, device_name="fedora-qa-bot")
        if isinstance(response, LoginResponse):
            print(f"Successfully logged in as {MATRIX_USER_ID}")
        else:
            print(f"Failed to log in: {response}")
            await self.client.close()
            exit(1)

    async def message_callback(self, room: MatrixRoom, event: RoomMessageText):
        if event.sender == self.client.user_id:
            return

        user_text = event.body.lower()
        bot_display_name = (await self.client.get_displayname(self.client.user_id)).displayname

        if bot_display_name.lower() not in user_text:
            return

        print(f"Received mention from {event.sender} in {room.display_name}: {event.body}")

        for keyword, answer in FEDORA_COREOS_QA.items():
            if keyword in user_text:
                response_text = f"🤖 (Quick Answer): {answer}"
                await self.send_message(room.room_id, response_text)
                return

        await self.send_message(room.room_id, "🤖 One moment, consulting the AI...")

        try:
            ai_prompt = f"""You are an expert virtual assistant specializing in Fedora CoreOS, containers, Podman, and related cloud-native technologies.
            You are the community-focused counterpart to the RHEL CoreOS expert.
            Your task is to answer the user's question clearly, technically, and didactically.
            If the question is not related to these IT topics, politely state that you have only been trained on subjects related to Fedora CoreOS.

            User's question: "{user_text}" """
            response = await self.ai_model.generate_content_async(ai_prompt)
            await self.send_message(room.room_id, response.text)
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            await self.send_message(room.room_id, "Sorry, an error occurred with the AI.")

    async def auto_join_invites(self, room: MatrixRoom, event: InviteMemberEvent):
        if event.state_key == self.client.user_id:
            print(f"Joining invited room: {room.room_id}")
            await self.client.join(room.room_id)

    async def send_message(self, room_id, message):
        await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": message}
        )
    
    async def run(self):
        await self.login()
        self.client.add_event_callback(self.message_callback, RoomMessageText)
        self.client.add_event_callback(self.auto_join_invites, InviteMemberEvent)
        print("Bot is running and listening for messages...")
        await self.client.sync_forever(timeout=30000)

if __name__ == "__main__":
    bot = MatrixBot()
    asyncio.run(bot.run())