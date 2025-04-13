# Show Me The Bunny: Easter Showbags Chat

An interactive chat application that allows users to explore and inquire about Easter showbags. This application uses AI to provide information about showbags, including prices, retail values, included items, and images.

## Features

- Chat with AI about Easter showbags
- View detailed information about showbags matching your query
- See images of showbags
- Get recommendations based on price, value, or content
- Find where to purchase specific showbags (stand numbers)

## Setup

1. Ensure you have Python 3.9+ installed on your system

2. Clone this repository:
   ```bash
   git clone <repository-url>
   cd show-me-the-bunny
   ```

3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

6. Run the application:
   ```bash
   python app.py
   ```

7. Open the provided URL in your browser (typically http://127.0.0.1:7860)

## Usage Examples

Here are some sample queries you can try:

- "What are the best value chocolate showbags?"
- "Show me showbags under $10"
- "Find showbags with toys for a 5-year-old girl"
- "Which showbags have the highest retail value?"
- "Where can I find the Bertie Beetle showbag?"
- "What items are included in the Harry Potter showbag?"

## Data Source

The application uses data from the `data/showbags/showbags.csv` file, which contains information about various Easter showbags.

## Dependencies

- gradio: For the web interface
- llama-index: For vector search and retrieval
- openai: For embeddings and language model
- pandas: For data processing
- python-dotenv: For environment variable management
