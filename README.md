# Gemini 2.5 Pro Code Analysis App

## Overview

This application enables users to upload files, including multiple files or even a ZIP archive containing an entire project, to a chat-based interface. Users can ask questions about their project, troubleshoot issues, or improve their codebase. Unlike traditional AI code editors that struggle with large contexts due to their limitations, Gemini 2.5 Pro, with its long context window, can effectively analyze and resolve issues across an entire project.


![image1](https://github.com/user-attachments/assets/b6631fb3-b662-48d4-89fa-dcf6cfb37d92)



## Features

- **Code Analysis**: Upload ZIP files containing code in various programming languages
- **Interactive Chat**: Ask questions about your code and receive detailed responses
- **Streaming Responses**: Get real-time responses as the AI generates them
- **Multi-file Support**: Analyze multiple files at once from a ZIP archive or upload multiple individual files
- **Context Preservation**: The AI maintains context of your code throughout the conversation
- **Support for Multiple Languages**: Handles various programming languages and file types
- **Reset Functionality**: Clear the conversation and uploaded files to start fresh

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

1. **Upload Code**: Click the "Upload" button to select a ZIP file or multiple individual code files.
2. **Ask Questions**: Type your questions about the code in the text input field.
3. **Get Responses**: The AI will analyze your code and respond to your questions.
4. **Reset**: Click the "Reset" button to clear the conversation and uploaded files when you want to start fresh.

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
- The application uses the `gemini-2.5-pro-exp-03-25` model version.
- Large code files may take longer to process.
- The application runs locally and does not store your code or conversations on external servers.
- The UI uses the Ocean theme from Gradio for a pleasant user experience.
