# Gemini 2.5 Pro Code Analysis App

## Overview

This application allows users to upload ZIP files containing code and chat with Google's Gemini 2.5 Pro AI model about the code. The AI can analyze the code, answer questions, and provide insights based on the uploaded files.

![Screenshot 2025-03-28 021114](https://github.com/user-attachments/assets/97ddfd21-cd8d-4b2f-9b71-e4780c8d6e02)


## Features

- **Code Analysis**: Upload ZIP files containing code in various programming languages
- **Interactive Chat**: Ask questions about your code and receive detailed responses
- **Streaming Responses**: Get real-time responses as the AI generates them
- **Multi-file Support**: Analyze multiple files at once from a ZIP archive
- **Context Preservation**: The AI maintains context of your code throughout the conversation
- **Support for Multiple Languages**: Handles various programming languages and file types

## Requirements

- Python 3.7+
- Google API Key for Gemini 2.5 Pro
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up your Google API Key as an environment variable:

```bash
# On Windows
set GOOGLE_API_KEY=your_api_key_here

# On macOS/Linux
export GOOGLE_API_KEY=your_api_key_here
```

## Usage

### Running the Application

Start the application by running:

```bash
python main.py
```

This will launch a Gradio web interface on `http://localhost:9595`.

### Analyzing Code

1. **Upload Code**: Click the "Upload" button to select a ZIP file containing your code files.
2. **Ask Questions**: Type your questions about the code in the text input field.
3. **Get Responses**: The AI will analyze your code and respond to your questions.

### Supported File Types

The application can extract and analyze text-based files with the following extensions:
- Python (.py)
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)
- HTML (.html)
- CSS (.css)
- Java (.java)
- C/C++ (.c, .cpp, .h)
- C# (.cs)
- PHP (.php)
- Ruby (.rb)
- Go (.go)
- Rust (.rs)
- Markdown (.md)
- Text (.txt)
- JSON (.json)
- XML (.xml)
- YAML (.yaml, .yml)
- Configuration files (.toml, .ini, .cfg, .conf)
- Shell scripts (.sh, .bat, .ps1)

## Project Structure

- `main.py`: The main application file with the Gradio UI and Gemini integration
- `requirements.txt`: List of required Python packages
- `notebook.ipynb`: Jupyter notebook for development and testing

## Dependencies

- `google-genai`: Google's Generative AI Python SDK
- `gradio`: Web interface framework for ML models
- `zipfile36`: Library for working with ZIP archives

## Notes

- The application requires a valid Google API Key with access to the Gemini 2.5 Pro model.
- Large code files may take longer to process.
- The application runs locally and does not store your code or conversations on external servers.
