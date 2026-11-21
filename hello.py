from dotenv import load_dotenv
import anthropic

load_dotenv()                      # reads .env into environment variables
client = anthropic.Anthropic()     # picks up ANTHROPIC_API_KEY automatically

msg = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a 6-line customer support call transcript about a billing error."}],
)

print(next(b.text for b in msg.content if b.type == "text"))
print(msg.usage)
