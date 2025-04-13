import gradio as gr
import pandas as pd
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from pathlib import Path
from dotenv import load_dotenv
import os
import json
from typing import List, Dict, Any

# Load environment variables from .env file (searching up the folder structure)
def find_dotenv_path():
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:  # Until we reach the root directory
        env_path = current_dir / '.env'
        if env_path.exists():
            return env_path
        current_dir = current_dir.parent
    return None

env_path = find_dotenv_path()
if env_path:
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()  # Try to load from the current directory as a fallback

# Initialize OpenAI embedding model with API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embed_model = OpenAIEmbedding()
Settings.embed_model = embed_model

# Path to the showbags CSV file
SHOWBAGS_PATH = Path("../../data/showbags/showbags_enriched.csv")

def load_showbags_index():
    """Load and process showbags data from CSV file"""
    try:
        # Check if the CSV file exists
        if not SHOWBAGS_PATH.exists():
            raise FileNotFoundError(f"Showbags data file not found at {SHOWBAGS_PATH}")
            
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(SHOWBAGS_PATH)
        
        # Format each showbag as a document with structured information
        documents = []
        
        for _, row in df.iterrows():
            # Convert the row to a dictionary for easier access
            showbag = row.to_dict()
            
            # Format the document text with key information
            text = f"""
            Showbag: {showbag.get('name', 'N/A')}
            Price: ${showbag.get('price_numeric', 'N/A')}
            Stand Numbers: {showbag.get('stand_numbers', 'N/A')}
            Image URL: {showbag.get('image_url', 'N/A')}
            Image Description: {showbag.get('img_description', 'N/A')}
            Retail Value: ${showbag.get('retail_value_numeric', 'N/A')}
            Distributor: {showbag.get('distributor', 'N/A')}
            
            Included Items:
            {showbag.get('included_items', 'N/A')}
            """
            
            # Create a Document object with the formatted text and metadata
            doc = Document(
                text=text,
                metadata={
                    "id": showbag.get('id', ''),
                    "name": showbag.get('name', ''),
                    "price": showbag.get('price_numeric', 0),
                    "image_url": showbag.get('image_url', ''),
                    "stand_numbers": showbag.get('stand_numbers', ''),
                    "retail_value": showbag.get('retail_value_numeric', 0),
                    "img_description": showbag.get('img_description', ''),
                }
            )
            documents.append(doc)
        
        # Create and return index from documents with sentence splitter
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        return VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            transformations=[parser]
        )
    
    except Exception as e:
        print(f"Error loading showbags data: {str(e)}")
        return None

def format_showbag_for_display(node):
    """Format a showbag node for display in the chat interface"""
    metadata = node.metadata or {}
    text = node.text or ""
    
    # Extract image URL from metadata or text
    image_url = metadata.get("image_url", "")
    if not image_url and "Image URL:" in text:
        image_url = text.split("Image URL:")[1].split("\n")[0].strip()
    
    # Format the display
    formatted = f"## {metadata.get('name', 'Showbag')}\n"
    formatted += f"**Price:** ${metadata.get('price', 'N/A')}\n"
    formatted += f"**Retail Value:** ${metadata.get('retail_value', 'N/A')}\n"
    formatted += f"**Stand Numbers:** {metadata.get('stand_numbers', 'N/A')}\n\n"
    
    if image_url:
        formatted += f"![Showbag Image]({image_url})\n\n"
    
    # Add the full text description with included items
    formatted += text.replace("Image URL:", "**Image URL:**")
    
    return formatted

def query_showbags(query, chat_history=None):
    """Query the showbags data based on user input"""
    try:
        # Get the pre-loaded index
        index = index_ref[0]
        if not index:
            return "Error: Could not load showbags data. Please check that the CSV file exists and is accessible."
        
        # Configure retriever for better context
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,  # Retrieve slightly more nodes initially to ensure we find unique ones
        )
        
        # Create query engine with custom retriever
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
        )
        
        # Query and get response
        response = query_engine.query(query)
        
        # Format the response for display, ensuring uniqueness
        formatted_response = "Here are some Easter showbags that match your query:\n\n"
        displayed_ids = set() # Keep track of displayed showbag IDs
        unique_showbags_count = 0
        max_display_showbags = 3 # Target number of unique showbags to display
        
        # Check if we have source nodes (retrieved showbags)
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                if unique_showbags_count >= max_display_showbags:
                    break # Stop once we have enough unique showbags
                    
                showbag_id = node.metadata.get('id')
                if showbag_id and showbag_id not in displayed_ids:
                    unique_showbags_count += 1
                    displayed_ids.add(showbag_id)
                    formatted_response += f"### Showbag {unique_showbags_count}:\n"
                    formatted_response += format_showbag_for_display(node)
                    formatted_response += "\n---\n\n"
            
            # Add the LLM response as a summary/recommendation if we found any bags
            if unique_showbags_count > 0:
                 formatted_response += f"**Summary:** {str(response)}"
            else:
                 formatted_response = f"Sorry, I couldn't find any specific showbags matching your query, but here's a general response:\n\n{str(response)}"

        else:
            # Fallback to just the response if no source nodes
            formatted_response += str(response)
        
        return formatted_response
    
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Reference to hold the index for reuse
index_ref = [None]

# Gradio interface setup
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Show Me The Bunny: Easter Showbags Chat")
    gr.Markdown("Ask questions about Easter showbags and get information, images, and recommendations!")
    
    with gr.Row():
        # Chat interface
        chatbot = gr.Chatbot(
            label="Conversation",
            height=600,
        )
    
    with gr.Row():
        # Query input
        with gr.Column(scale=4):
            msg = gr.Textbox(
                label="Ask about Easter showbags",
                placeholder="Example: What are the best value chocolate showbags?",
                lines=2
            )
        
        with gr.Column(scale=1):
            # Submit button
            submit_btn = gr.Button("Send", variant="primary")
    
    # Chat history state
    chat_state = gr.State([])
    
    def respond(message, chat_history):
        """Process user message and update chat history"""
        if not message.strip():
            return "", chat_history
            
        # Get response from query engine
        bot_response = query_showbags(message, chat_history)
        
        # Update chat history
        chat_history.append((message, bot_response))
        
        return "", chat_history
    
    # Set up interaction
    submit_btn.click(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    gr.Markdown("""
    ### Tips:
    - Ask about specific types of showbags (chocolate, toys, etc.)
    - Search by price range ("showbags under $10")
    - Find the best value showbags compared to retail price
    - Ask for recommendations for children, teens, or adults
    - Inquire about specific brands or characters
    - Find out where to buy specific showbags (stand numbers)
    """)

if __name__ == "__main__":
    # Check for environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables")
        print("Please make sure you have a valid OpenAI API key in your .env file")
    
    # Verify showbags file exists
    if not SHOWBAGS_PATH.exists():
        print(f"Warning: Showbags CSV file not found at {SHOWBAGS_PATH}")
        print(f"Please ensure the file exists at {SHOWBAGS_PATH.absolute()}")
    else:
        print(f"Showbags CSV file found at {SHOWBAGS_PATH}")
    
    # Load index on startup for faster initial query
    print("Loading showbags data into vector index...")
    index_ref[0] = load_showbags_index()
    if index_ref[0]:
        print("Index loaded successfully!")
    else:
        print("Failed to load showbags index. Please check the CSV file and try again.")
    
    # Launch the app
    app.launch() 