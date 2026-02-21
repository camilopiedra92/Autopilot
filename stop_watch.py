import asyncio
from autopilot.connectors import GmailConnector

async def main():
    print("Initializing GmailConnector...")
    gmail = GmailConnector()
    print("Calling stop()...")
    gmail.service.users().stop(userId="me").execute()
    print("Watch stopped successfully!")

if __name__ == "__main__":
    asyncio.run(main())
