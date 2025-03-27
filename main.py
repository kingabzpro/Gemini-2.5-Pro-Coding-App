import os
import zipfile
from typing import Dict, List, Optional, Union

import gradio as gr
from google import genai
from google.genai import types

# Retrieve API key for Google GenAI from the environment variables.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize the client so that it can be reused across functions.
CLIENT = genai.Client(api_key=GOOGLE_API_KEY)

# General constants for the UI
TITLE = """<h1 align="center">✨ Gemini Software Engineer</h1>"""
AVATAR_IMAGES = (None, "https://media.roboflow.com/spaces/gemini-icon.png")


def extract_text_from_zip(zip_file_path: str) -> Dict[str, str]:
    """
    Extract text content from files in a ZIP archive.

    Parameters:
        zip_file_path (str): Path to the ZIP file.

    Returns:
        Dict[str, str]: Dictionary mapping filenames to their text content.
    """
    text_contents = {}

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            # Skip directories
            if file_info.filename.endswith("/"):
                continue

            # Skip binary files and focus on text files
            file_ext = os.path.splitext(file_info.filename)[1].lower()
            text_extensions = [
                ".py",
                ".js",
                ".html",
                ".css",
                ".java",
                ".c",
                ".cpp",
                ".h",
                ".cs",
                ".php",
                ".rb",
                ".go",
                ".rs",
                ".ts",
                ".jsx",
                ".tsx",
                ".md",
                ".txt",
                ".json",
                ".xml",
                ".yaml",
                ".yml",
                ".toml",
                ".ini",
                ".cfg",
                ".conf",
                ".sh",
                ".bat",
                ".ps1",
            ]

            if file_ext in text_extensions:
                try:
                    with zip_ref.open(file_info) as file:
                        content = file.read().decode("utf-8", errors="replace")
                        text_contents[file_info.filename] = content
                except Exception as e:
                    text_contents[file_info.filename] = (
                        f"Error extracting file: {str(e)}"
                    )

    return text_contents


# Global variables
EXTRACTED_FILES = {}

# Store chat sessions
CHAT_SESSIONS = {}


def upload_zip(files: Optional[List[str]], chatbot: List[Union[dict, gr.ChatMessage]]):
    """
    Process uploaded ZIP files: extract text content and append a message to the chat.

    Parameters:
        files (Optional[List[str]]): List of ZIP file paths.
        chatbot (List[Union[dict, gr.ChatMessage]]): The conversation history.

    Returns:
        List[Union[dict, gr.ChatMessage]]: Updated conversation history.
    """
    global EXTRACTED_FILES

    for file in files:
        filename = os.path.basename(file)
        extracted_files = extract_text_from_zip(file)

        if not extracted_files:
            chatbot.append(
                gr.ChatMessage(
                    role="user",
                    content=f"<p>📦 ZIP file uploaded: {filename}, but no text files were found.</p>",
                )
            )
        else:
            file_list = "\n".join([f"- {name}" for name in extracted_files.keys()])
            chatbot.append(
                gr.ChatMessage(
                    role="user",
                    content=f"<p>📦 ZIP file uploaded: {filename}</p><p>Extracted {len(extracted_files)} text files:</p><pre>{file_list}</pre>",
                )
            )

            # Store the extracted content in the global variable
            EXTRACTED_FILES[filename] = extracted_files

    return chatbot


def user(text_prompt: str, chatbot: List[gr.ChatMessage]):
    """
    Append a new user text message to the chat history.

    Parameters:
        text_prompt (str): The input text provided by the user.
        chatbot (List[gr.ChatMessage]): The existing conversation history.

    Returns:
        Tuple[str, List[gr.ChatMessage]]: A tuple of an empty string (clearing the prompt)
            and the updated conversation history.
    """
    if text_prompt:
        chatbot.append(gr.ChatMessage(role="user", content=text_prompt))
    return "", chatbot


def get_message_content(msg):
    """
    Retrieve the content of a message that can be either a dictionary or a gr.ChatMessage.

    Parameters:
        msg (Union[dict, gr.ChatMessage]): The message object.

    Returns:
        str: The textual content of the message.
    """
    if isinstance(msg, dict):
        return msg.get("content", "")
    return msg.content


def send_to_gemini(chatbot: List[Union[dict, gr.ChatMessage]]):
    """
    Send the user's prompt to Gemini and display the response.
    If code files were uploaded, they will be included in the context.

    Parameters:
        chatbot (List[Union[dict, gr.ChatMessage]]): The conversation history.

    Returns:
        List[Union[dict, gr.ChatMessage]]: The updated conversation history with Gemini's response.
    """
    global EXTRACTED_FILES, CHAT_SESSIONS

    if len(chatbot) == 0:
        chatbot.append(
            gr.ChatMessage(
                role="assistant",
                content="Please enter a message to start the conversation.",
            )
        )
        return chatbot

    # Get the last user message as the prompt
    user_messages = [
        msg
        for msg in chatbot
        if (isinstance(msg, dict) and msg.get("role") == "user")
        or (hasattr(msg, "role") and msg.role == "user")
    ]

    if not user_messages:
        chatbot.append(
            gr.ChatMessage(
                role="assistant",
                content="Please enter a message to start the conversation.",
            )
        )
        return chatbot

    last_user_msg = user_messages[-1]
    prompt = get_message_content(last_user_msg)

    # Skip if the last message was about uploading a ZIP file
    if "📦 ZIP file uploaded:" in prompt:
        chatbot.append(
            gr.ChatMessage(
                role="assistant",
                content="What would you like to know about the code in this ZIP file?",
            )
        )
        return chatbot

    # Generate a unique session ID based on the extracted files or use a default key for no files
    if EXTRACTED_FILES:
        session_key = ",".join(sorted(EXTRACTED_FILES.keys()))
    else:
        session_key = "no_files"

    # Create a new chat session if one doesn't exist for this set of files
    if session_key not in CHAT_SESSIONS:
        # Configure Gemini with code execution capability
        CHAT_SESSIONS[session_key] = CLIENT.chats.create(
            model="gemini-2.5-pro-exp-03-25",
        )

        # Send all extracted files to the chat session first
        initial_contents = []
        for zip_name, files in EXTRACTED_FILES.items():
            for filename, content in files.items():
                file_ext = os.path.splitext(filename)[1].lower()
                mime_type = "text/plain"

                # Set appropriate mime type based on file extension
                if file_ext == ".py":
                    mime_type = "text/x-python"
                elif file_ext in [".js", ".jsx"]:
                    mime_type = "text/javascript"
                elif file_ext in [".ts", ".tsx"]:
                    mime_type = "text/typescript"
                elif file_ext == ".html":
                    mime_type = "text/html"
                elif file_ext == ".css":
                    mime_type = "text/css"
                elif file_ext in [".json", ".jsonl"]:
                    mime_type = "application/json"
                elif file_ext in [".xml", ".svg"]:
                    mime_type = "application/xml"

                # Create a header with the filename to preserve original file identity
                file_header = f"File: {filename}\n\n"
                file_content = file_header + content

                initial_contents.append(
                    types.Part.from_bytes(
                        data=file_content.encode("utf-8"),
                        mime_type=mime_type,
                    )
                )

        # Initialize the chat context with files if available
        if initial_contents:
            initial_contents.append(
                "I've uploaded these code files for you to analyze. I'll ask questions about them next."
            )
            # Use synchronous API instead of async
            CHAT_SESSIONS[session_key].send_message(initial_contents)
        # For sessions without files, we don't need to send an initial message

    # Append a placeholder for the assistant's response
    chatbot.append(gr.ChatMessage(role="assistant", content=""))

    # Send the user's prompt to the existing chat session using streaming API
    response = CHAT_SESSIONS[session_key].send_message_stream(prompt)

    # Process the response stream - text only (no images)
    output_text = ""
    for chunk in response:
        if chunk.candidates and chunk.candidates[0].content.parts:
            for part in chunk.candidates[0].content.parts:
                if part.text is not None:
                    # Append the new chunk of text
                    output_text += part.text

                    # Update the last assistant message with the current accumulated response
                    if isinstance(chatbot[-1], dict):
                        chatbot[-1]["content"] = output_text
                    else:
                        chatbot[-1].content = output_text

                    # Yield the updated chatbot to show streaming updates in the UI
                    yield chatbot

    # Return the final chatbot state
    return chatbot


# Define the Gradio UI components
chatbot_component = gr.Chatbot(
    label="Gemini 2.5 Pro",
    type="messages",
    bubble_full_width=False,
    avatar_images=AVATAR_IMAGES,
    scale=2,
    height=350,
)
text_prompt_component = gr.Textbox(
    placeholder="Ask a question or upload code files to analyze...",
    show_label=False,
    autofocus=True,
    scale=28,
)
upload_zip_button_component = gr.UploadButton(
    label="Upload",
    file_count="multiple",
    file_types=[".zip"],
    scale=1,
    min_width=80,
)
send_button_component = gr.Button(
    value="Send", variant="primary", scale=0.3, min_width=80
)

# Define input lists for button chaining
user_inputs = [text_prompt_component, chatbot_component]

with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.HTML(TITLE)
    with gr.Column():
        chatbot_component.render()
        with gr.Row(equal_height=True):
            text_prompt_component.render()
            send_button_component.render()
            upload_zip_button_component.render()

    # When the Send button is clicked, first process the user text then send to Gemini
    send_button_component.click(
        fn=user,
        inputs=user_inputs,
        outputs=[text_prompt_component, chatbot_component],
        queue=False,
    ).then(
        fn=send_to_gemini,
        inputs=[chatbot_component],
        outputs=[chatbot_component],
        api_name="send_to_gemini",
    )

    # Allow submission using the Enter key
    text_prompt_component.submit(
        fn=user,
        inputs=user_inputs,
        outputs=[text_prompt_component, chatbot_component],
        queue=False,
    ).then(
        fn=send_to_gemini,
        inputs=[chatbot_component],
        outputs=[chatbot_component],
        api_name="send_to_gemini_submit",
    )

    # Handle ZIP file uploads
    upload_zip_button_component.upload(
        fn=upload_zip,
        inputs=[upload_zip_button_component, chatbot_component],
        outputs=[chatbot_component],
        queue=False,
    )

# Launch the demo interface
demo.queue(max_size=99, api_open=False).launch(
    debug=False,
    show_error=True,
    server_port=9595,
    server_name="localhost",
)
