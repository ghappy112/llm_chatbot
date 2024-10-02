from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
import time

client = InferenceClient(token="hf_token")

app = Flask(__name__)

chat_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Chatbot</title>
                <style>
                    body {
                        font-family: 'Arial', sans-serif;
                        background-color: #f4f4f9;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                    }
                    .chat-container {
                        width: 500px;
                        background-color: #fff;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        border-radius: 8px;
                        overflow: hidden;
                    }
                    .chat-header {
                        background-color: #4CAF50;
                        color: white;
                        padding: 10px;
                        text-align: center;
                        font-size: 1.2em;
                    }
                    .chat-box {
                        padding: 10px;
                        height: 300px;
                        overflow-y: scroll;
                        border-bottom: 1px solid #ddd;
                    }
                    .input-box {
                        display: flex;
                        padding: 10px;
                        border-top: 1px solid #ddd;
                    }
                    .input-box input {
                        flex: 1;
                        padding: 10px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        margin-right: 10px;
                    }
                    .input-box button {
                        padding: 10px 20px;
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    }
                    .input-box button:hover {
                        background-color: #45a049;
                    }
                    .button-box {
                        text-align: center;
                        padding: 10px;
                    }
                    .button-box button {
                        padding: 10px 20px;
                        background-color: #f44336;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    }
                    .button-box button:hover {
                        background-color: #e53935;
                    }
                    .chat-box div {
                        margin-bottom: 10px;
                    }
                    .chat-box div strong {
                        color: #333;
                    }
                </style>
            </head>
            <body>
                <div class="chat-container">
                    <div class="chat-header">Chatbot</div>
                    <div class="chat-box" id="chat-box"></div>
                    <div class="input-box">
                        <input type="text" id="user-input" placeholder="Type your message here..." />
                        <button onclick="sendMessage()">Send</button>
                    </div>
                    <div class="button-box">
                        <button onclick="clearChat()">Clear Chat</button>
                    </div>
                </div>

                <script>
                    function sendMessage() {
                        const userInput = document.getElementById('user-input').value;
                        if (userInput.trim() === "") return;

                        const chatBox = document.getElementById('chat-box');
                        chatBox.innerHTML += `<div><strong>You:</strong> ${userInput}</div>`;

                        fetch('/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ message: userInput })
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.response) {
                                chatBox.innerHTML += `<div><strong>Bot:</strong> ${data.response}</div>`;
                            } else {
                                chatBox.innerHTML += `<div><strong>Error:</strong> ${data.error}</div>`;
                            }
                            chatBox.scrollTop = chatBox.scrollHeight;
                        });

                        document.getElementById('user-input').value = "";
                    }

                    function clearChat() {
                        fetch('/clear', {
                            method: 'POST'
                        })
                        .then(() => {
                            document.getElementById('chat-box').innerHTML = "";
                        });
                    }
                </script>
            </body>
            </html>
            """

def llm_chat(messages, model="mistralai/Mistral-7B-Instruct-v0.3"):
    conversation = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        conversation += f"<{role}>{content}</{role}>"
    
    response = client.text_generation(
        model=model,
        prompt=conversation,
        max_new_tokens=500,
        stream=True,
    )

    messages = []
    for message in response:
        messages.append(message)

    while messages[0] == '\n':
        messages = messages[1:]

    text = ''.join(messages).replace(' . ', '. ').replace(' , ',  ', ').replace(" - ", "-").replace("<assistant>", "").replace("</assistant>", "").replace("</s>", "").replace("<s>", "").strip()

    if text[:5] == "<user>" and "</user>" in text:
        text = text[text.index("</user>") + len("</user>"):]
    if "<user>" in text:
        text = text[:text.index("<user>")]
    text = text.replace("<user>", "").replace("</user>", "")
    
    return text

@app.route('/')
def home():
    global conversation_history
    conversation_history = [
        {"role": "sys", "content": "You are a helpful and intelligent chatbot."}
    ]
    return chat_html

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history
    user_message = request.json.get("message")
    conversation_history.append({"role": "user", "content": user_message})
    try:
        response = llm_chat(conversation_history)
        conversation_history.append({"role": "assistant", "content": response})
        return jsonify({"response": response})
    except Exception as e:
        error_message = str(e).lower()
        if "rate limit reached" in error_message or "too many requests" in error_message:
            return jsonify({"error": "Rate limit reached. Please wait a few hours and try again."})
        else:
            tries = 0
            while tries < 10:
                time.sleep(1)
                conversation_history.append({"role": "user", "content": "?"})
                try:
                    response = llm_chat(conversation_history)
                    conversation_history.append({"role": "assistant", "content": response})
                    return jsonify({"response": response})
                except Exception as e:
                    tries += 1
                    error_message = str(e).lower()
                    if "rate limit reached" in error_message or "too many requests" in error_message:
                        return jsonify({"error": "Rate limit reached. Please wait a few hours and try again."})
            return jsonify({"error": "Internal Error, please try again."})

@app.route('/clear', methods=['POST'])
def clear():
    global conversation_history
    conversation_history = [
        {"role": "sys", "content": "You are a helpful and intelligent chatbot."}
    ]
    return '', 204

if __name__ == "__main__":
    app.run(debug=True)
