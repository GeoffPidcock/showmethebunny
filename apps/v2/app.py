import gradio as gr
import pandas as pd
# Updated LlamaIndex imports for modular packages
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
import sqlite3  # Add sqlite3 import
from datetime import datetime  # Add datetime import
import uuid  # Add uuid import

# Load environment variables from .env file (searching up the folder structure)
def find_dotenv_path():
    current_dir = Path.cwd()
    # Avoid searchingsensitive/private data
    while current_dir != current_dir.parent and "sensitive" not in str(current_dir).lower() and "private" not in str(current_dir).lower(): # Until we reach the root directory
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

# --- Configuration ---
# Use environment variables for paths, falling back to defaults
DB_FILE = os.getenv("DB_PATH", "showbag_logs.db")
# Default path relative to this file's location
DEFAULT_CSV_PATH = Path(__file__).parent.parent.parent / "data" / "showbags" / "showbags_enriched.csv"
SHOWBAGS_PATH = Path(os.getenv("CSV_PATH", DEFAULT_CSV_PATH))
# API Key setup MUST happen after potential load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables.")
else:
    # Set it for llama-index/openai libraries to find
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# --- End Configuration ---

# --- Database Setup ---
def init_db(db_path=DB_FILE):
    """Initialize the SQLite database and create tables if they don't exist."""
    try:
        # Ensure the directory for the database exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True) # Create parent directories if needed

        conn = sqlite3.connect(db_path) # Use the provided or default path
        c = conn.cursor()
        # Create interactions table
        c.execute('''CREATE TABLE IF NOT EXISTS interactions (
                     id TEXT PRIMARY KEY,
                     timestamp TEXT,
                     user_query TEXT,
                     retrieved_context TEXT,  -- Store as JSON string
                     final_response TEXT
                 )''')
        # Create votes table
        c.execute('''CREATE TABLE IF NOT EXISTS votes (
                     id TEXT PRIMARY KEY,
                     timestamp TEXT,
                     showbag_id TEXT, -- Identifier for the showbag (e.g., name or unique ID from CSV)
                     vote_type TEXT -- 'up' or 'down'
                 )''')
        conn.commit()
        conn.close()
        print(f"Database initialized successfully at {db_path}") # Log the path used
    except Exception as e:
        print(f"Error initializing database: {e}")

# --- LlamaIndex Setup ---
# Initialize OpenAI embedding model (ensure API key is set)
try:
    # Modern initialization with the modular packages
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.embed_model = embed_model
    print("OpenAI embedding model initialized successfully")
except Exception as e:
    print(f"Error initializing OpenAI embedding model: {e}")
    # Fallback to a stub embedding model if needed
    class DummyEmbedding:
        def __call__(self, texts):
            return [[0.0] * 768 for _ in texts]  # Return stub embeddings
    
    Settings.embed_model = DummyEmbedding()
    print("Using dummy embedding model due to initialization error")

# --- Data Loading ---
def load_showbags_data(csv_path=SHOWBAGS_PATH):
    """Load showbags DataFrame and create LlamaIndex."""
    print(f"Attempting to load showbags data from: {csv_path}") # Add logging
    try:
        # Check if the CSV file exists
        if not Path(csv_path).exists(): # Use the provided path
            raise FileNotFoundError(f"Showbags data file not found at {csv_path}")

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path) # Use the provided path

        # --- LlamaIndex Creation ---
        documents = []
        skipped_count = 0
        for index, row in df.iterrows(): # Use index for potential logging
            showbag = row.to_dict()
            
            # --- Get and Validate ID --- 
            doc_id = showbag.get('id')
            # Check if ID is missing, null, or empty string after stripping whitespace
            if pd.isna(doc_id) or not str(doc_id).strip():
                print(f"Warning: Skipping showbag '{showbag.get('name', f'at index {index}')}' due to missing or invalid ID: {doc_id}")
                skipped_count += 1
                continue # Skip this row
            # Ensure ID is a string for consistency in metadata and filtering
            doc_id = str(doc_id).strip() 
            # --- End ID Validation --- 

            # Format the document text (as before)
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
            
            # Create a Document object - Use the validated doc_id
            doc = Document(
                text=text,
                metadata={
                    # Use the validated ID from CSV, no fallback
                    "id": doc_id, 
                    "name": showbag.get('name', ''),
                    "price": showbag.get('price_numeric', 0),
                    "image_url": showbag.get('image_url', ''),
                    "stand_numbers": showbag.get('stand_numbers', ''),
                    "retail_value": showbag.get('retail_value_numeric', 0),
                    "img_description": showbag.get('img_description', ''),
                }
            )
            documents.append(doc)
        
        if skipped_count > 0:
             print(f"Note: Skipped {skipped_count} showbags during indexing due to missing/invalid IDs.")
        
        if not documents:
             print("Error: No valid documents found to index after filtering for IDs.")
             return None, df # Return df but no index

        # Create index from documents
        parser = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            transformations=[parser]
        )
        # --- End LlamaIndex Creation ---

        print(f"Loaded {len(df)} showbags and created index for {len(documents)} valid documents.")
        return index, df
    
    except Exception as e:
        print(f"Error loading showbags data: {str(e)}")
        return None, None

# --- Logging Functions ---
def log_interaction(user_query: str, retrieved_context: str, final_response: str):
    """Log interaction details to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_FILE) # DB_FILE uses the env var or default
        c = conn.cursor()
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        c.execute("INSERT INTO interactions (id, timestamp, user_query, retrieved_context, final_response) VALUES (?, ?, ?, ?, ?)",
                  (interaction_id, timestamp, user_query, retrieved_context, final_response))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging interaction: {e}")

def log_vote(showbag_id: str, vote_type: str):
    """Log a vote to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_FILE) # DB_FILE uses the env var or default
        c = conn.cursor()
        vote_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        c.execute("INSERT INTO votes (id, timestamp, showbag_id, vote_type) VALUES (?, ?, ?, ?)",
                  (vote_id, timestamp, showbag_id, vote_type))
        conn.commit()
        conn.close()
        print(f"Logged vote: {vote_type} for {showbag_id}")
    except Exception as e:
        print(f"Error logging vote: {e}")

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
    """Query the showbags data based on user input (using LlamaIndex)"""
    try:
        # Get the pre-loaded index and dataframe from app_data
        index = app_data.get("index")
        if not index:
            return "Error: Showbags index not loaded. Please wait for initialization."
        # Get the dataframe for potential use (though not directly used in this query function)
        # df = app_data.get("df")

        # Configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,
        )
        
        # Create query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
        )
        
        # Query and get response
        response = query_engine.query(query)
        
        # Prepare context for logging
        retrieved_context_data = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node_with_score in response.source_nodes:
                node_info = {
                    "id": node_with_score.metadata.get('id', ''),
                    "name": node_with_score.metadata.get('name', ''),
                    "score": node_with_score.score,
                    # Optionally add a snippet of text if needed
                    # "text_snippet": node_with_score.text[:100] + "..."
                }
                retrieved_context_data.append(node_info)
        retrieved_context_json = json.dumps(retrieved_context_data)

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
        
        # Log the interaction before returning
        try:
            log_interaction(user_query=query, retrieved_context=retrieved_context_json, final_response=formatted_response)
        except Exception as log_e:
            print(f"Logging failed: {log_e}") 

        return formatted_response
    
    except Exception as e:
        return f"Error processing query: {str(e)}"

# --- Vote Handling Function ---
def handle_vote(vote_type, showbag_id, current_shortlist):
    """Handles the like/dislike button clicks, logs the vote, and updates the shortlist."""
    if not showbag_id:
        return "Please select a showbag before voting.", current_shortlist # Return unchanged shortlist
    
    log_vote(showbag_id, vote_type)
    
    if current_shortlist is None:
        current_shortlist = []

    new_shortlist = list(current_shortlist)
    message = f"Vote ({vote_type}) recorded."

    df = app_data.get("df") # Get df from global app_data

    if vote_type == 'up' and df is not None:
        selected_row_df = df[df['id'] == showbag_id]
        if not selected_row_df.empty:
            selected_row = selected_row_df.iloc[0]
            name = selected_row.get('name', 'N/A')
            price = selected_row.get('price_numeric', 'N/A')
            stands = selected_row.get('stand_numbers', 'N/A')
            try:
                price_str = f"${price:.2f}" if pd.notna(price) else "N/A"
            except TypeError:
                price_str = "N/A"
            shortlist_item_str = f"- {name} ({price_str}) - Stands: {stands}"
            already_exists = any(shortlist_item_str == item for item in new_shortlist)
            if not already_exists:
                new_shortlist.append(shortlist_item_str)
                message += f" Added '{name}' to shortlist!"
            else:
                message += f" '{name}' is already in the shortlist."
        else:
            message += " Could not find showbag details to add to shortlist."
    elif vote_type == 'up' and df is None:
         message += " Error accessing showbag data for shortlist."

    return message, new_shortlist

# --- Display/Filtering Functions ---
def search_and_display(search_query: str, max_price: float, df: pd.DataFrame, cached_ids: List[str] | None = None):
    """Filters DataFrame based on semantic search (if query provided), price, and optional cached IDs."""
    index = app_data.get("index") # Get index from global app_data
    if df is None: return [], "Error: Showbag data not loaded.", None, None
    if index is None: return [], "Error: Showbag index not loaded.", None, None

    SIMILARITY_THRESHOLD = 0.75
    df['price_numeric'] = pd.to_numeric(df['price_numeric'], errors='coerce')
    df['id'] = df['id'].fillna('').astype(str)
    working_df = df.copy()
    search_performed = False
    top_result_id = None
    current_ordered_ids = None

    if search_query and search_query.strip():
        search_performed = True
        try:
            retriever = VectorIndexRetriever(index=index, similarity_top_k=50)
            retrieved_nodes_with_scores = retriever.retrieve(search_query)
            high_similarity_nodes = [n for n in retrieved_nodes_with_scores if n.score >= SIMILARITY_THRESHOLD]
            current_ordered_ids = list(dict.fromkeys([str(node.metadata['id']) for node in high_similarity_nodes if 'id' in node.metadata]))
            print(f"Search found {len(current_ordered_ids)} unique IDs above threshold.")
        except Exception as e:
            print(f"Error during semantic search: {e}")
            current_ordered_ids = None
            return [], f"Error during search: {e}", None, current_ordered_ids
    else:
        current_ordered_ids = cached_ids
        print(f"No search query. Using cached IDs: {current_ordered_ids}")

    if current_ordered_ids:
        if current_ordered_ids:
            search_performed = True
            working_df = working_df[working_df['id'].isin(current_ordered_ids)]
            if not working_df.empty:
                 working_df['id'] = pd.Categorical(working_df['id'], categories=current_ordered_ids, ordered=True)
                 working_df = working_df.sort_values('id')

    final_filtered_df = working_df[working_df['price_numeric'] <= max_price].copy()

    if not final_filtered_df.empty:
        top_result_id = final_filtered_df.iloc[0]['id']

    gallery_data = []
    for _, row in final_filtered_df.iterrows():
        caption = f"{row.get('name', 'N/A')} - ${row.get('price_numeric', 'N/A'):.2f}"
        image_url = row.get('image_url', None)
        if pd.isna(image_url):
            image_url = None
        gallery_data.append((image_url, caption))

    num_results = len(gallery_data)
    if num_results > 0:
        if search_performed:
             query_part = f" matching '{search_query}'" if (search_query and search_query.strip()) else " from your filter"
             message = f"Found {num_results} showbags{query_part} below ${max_price:.0f}."
        else:
             message = f"Showing {num_results} showbags below ${max_price:.0f}."
    else:
        if search_performed:
            query_part = f" matching '{search_query}'" if (search_query and search_query.strip()) else " from your filter"
            message = f"No showbags found{query_part} below ${max_price:.0f}."
        else:
            message = f"No showbags found below ${max_price:.0f}."

    print(f"Returning {num_results} items for gallery. Message: '{message}'. Top ID: {top_result_id}. Filter IDs: {current_ordered_ids}")
    return gallery_data, message, top_result_id, current_ordered_ids

def get_selected_showbag_info(evt: gr.SelectData, df: pd.DataFrame):
    """Get the ID and display info of the selected showbag based on the gallery selection event."""
    if df is None or evt is None:
        return "Select a showbag to see details.", None
    try:
        selected_caption = evt.value.get('caption')
        if not selected_caption or not isinstance(selected_caption, str):
             print(f"Error: Could not extract caption string from evt.value: {evt.value}")
             return "Error extracting caption from selected item.", None
        name_part = selected_caption.split(' - $')[0].strip()
        matching_rows = df.loc[df['name'] == name_part]
        if matching_rows.empty:
            print(f"Error: Could not find showbag with name: {name_part}")
            return f"Could not find details for '{name_part}'", None
        selected_row = matching_rows.iloc[0]
        showbag_id = selected_row.get('id')
        if pd.isna(showbag_id) or not str(showbag_id).strip():
            showbag_id = None
            print(f"Warning: Missing or invalid ID for showbag '{name_part}'. Voting may not work correctly.")
        else:
            showbag_id = str(showbag_id).strip()
        name = selected_row.get('name', 'N/A')
        price = selected_row.get('price_numeric', 'N/A')
        retail_value = selected_row.get('retail_value_numeric', 'N/A')
        stands = selected_row.get('stand_numbers', 'N/A')
        included_items = selected_row.get('included_items', 'N/A')
        try:
            price_str = f"${price:.2f}" if pd.notna(price) else "N/A"
        except TypeError:
            price_str = "N/A"
        try:
            retail_value_str = f"${retail_value:.2f}" if pd.notna(retail_value) else "N/A"
        except TypeError:
            retail_value_str = "N/A"
        details = f"**{name}**\nPrice: {price_str}\nRetail Value: {retail_value_str}\nStands: {stands}\n\n**Includes:**\n{included_items}"
        print(f"Selected: {name}, ID: {showbag_id}")
        return details, showbag_id
    except Exception as e:
        print(f"Error getting selected info: {e}, Event Data: {evt}")
        error_val_display = selected_caption if 'selected_caption' in locals() else 'selected item'
        return f"Error displaying details for: {error_val_display}", None

# --- App Data Holder (Simplified - only holds runtime data if needed by handlers) ---
# UI instance will be a top-level variable now. Index/df are loaded directly below.
app_data = {}

# --- Initialize DB and Load Data AT MODULE LEVEL ---
print(f"app.py: Initializing DB using path: {DB_FILE}")
init_db() # Uses DB_FILE which reads env var DB_PATH or defaults

print(f"app.py: Loading data using path: {SHOWBAGS_PATH}")
# Load index and df into top-level variables for UI build
# These will be populated when the module loads.
showbags_index, showbags_df = load_showbags_data() # Uses SHOWBAGS_PATH which reads CSV_PATH or defaults

# Store loaded data in app_data for handler access if needed (optional, depends on handler implementation)
if showbags_index is not None:
    app_data['index'] = showbags_index
if showbags_df is not None:
    app_data['df'] = showbags_df

# --- UI Building Function ---
def build_ui(df_loaded, index_loaded):
    """Builds the Gradio UI interface."""
    print("Building Gradio UI...")
    # Store loaded data globally for access by UI event handlers
    # This is a simple approach; alternatives exist (e.g., class-based app)
    # Ensure event handlers use app_data if they need index/df after build
    app_data["df"] = df_loaded
    app_data["index"] = index_loaded

    # Check if data loading was successful during the build phase
    if df_loaded is None or index_loaded is None:
        print("Error: DataFrame or Index is None passed to build_ui.")
        # Return a simple UI indicating the error
        with gr.Blocks() as error_app:
            gr.Markdown("# Error Loading Application Data")
            gr.Markdown("Could not load the necessary showbags data or index to build the UI.")
        return error_app

    # Calculate total showbags for display
    total_showbags = len(df_loaded) if df_loaded is not None else 0
    initial_shortlist = [] # Start with an empty shortlist

    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown(f"""
        # 🐰 Show Me the Bunny! 🥕 Easter Showbag Finder ({total_showbags} bags)
        Ask about the Easter showbags you're interested in!
        """)
        gr.Markdown("Look for your favourite showbags below, or search for something special!")

        # --- Top Controls Row ---
        with gr.Row():
            with gr.Column(scale=4): # Wider column for search + examples
                with gr.Row(): # Nested row for search box and button
                    search_box = gr.Textbox(label="What showbag are you dreaming of?", 
                                             placeholder="Type here and press Enter or click Search...", # Updated placeholder
                                             scale=4)
                    search_button = gr.Button("Search", scale=1)
                
                # Examples below search box
                example_searches = [
                    "toddler boy who likes dinosaurs",
                    "expecting parents who are after a bargain",
                    "kid who loves bluey",
                    "something pink and sparkly for a 5 year old girl",
                    "best value chocolate bags",
                    "bags with dress up items"
                ]
                gr.Examples(examples=example_searches, 
                            inputs=search_box, 
                            label="Click an example to search!")
                            
            with gr.Column(scale=2): # Narrower column for slider + status
                 # Define the slider with an initial label
                price_slider = gr.Slider(minimum=0, maximum=50, value=50, step=1, label=f"Show bags cheaper than $50")
                status_message = gr.Markdown("")

        # --- Gallery Row ---
        gallery = gr.Gallery(label="Matching Showbags", 
                             elem_id="showbag_gallery", 
                             columns=[5], 
                             height=600, 
                             object_fit="contain",
                             preview=True)

        # --- Bottom Row: Details and Shortlist ---
        with gr.Row():
            with gr.Column(scale=2): # Details column
                gr.Markdown("**Selected Showbag Details**")
                selected_info = gr.Markdown("Click a showbag image above to see details.")
                
            with gr.Column(scale=2): # Shortlist column
                gr.Markdown("**My Shortlist**")
                # Hidden state to store the ID of the selected showbag for voting
                selected_showbag_id_state = gr.State(value=None)
                with gr.Row():
                    like_button = gr.Button("👍 Like")
                    dislike_button = gr.Button("👎 Dislike")
                vote_confirm_message = gr.Markdown("")
                shortlist_state = gr.State(value=[]) 
                # Make textbox interactive for easy copying
                shortlist_display = gr.Textbox(label="Your Shortlist (Select and copy text):", 
                                               lines=5, 
                                               interactive=True) # Set interactive=True

        # State to potentially hold the ID of the top search result for auto-selection attempt
        top_result_id_state = gr.State(value=None)
        last_search_ids_state = gr.State(value=None) # State for cached IDs

        # --- Event Handling ---
        # Wrapper for search_and_display call
        def update_gallery_wrapper(query, price, cached_ids):
            # Call the search/display logic
            gallery_data, msg, top_id, new_ids = search_and_display(query, price, df_loaded, cached_ids)
            
            # --- Log the interaction ---
            try:
                # Format context as JSON string (using the returned IDs)
                context_to_log = json.dumps(new_ids if new_ids is not None else [])
                # Use the status message as the final response logged
                log_interaction(user_query=query, retrieved_context=context_to_log, final_response=msg)
                print(f"Logged interaction: Query='{query}', ResponseMsg='{msg}'")
            except Exception as log_e:
                print(f"Error during interaction logging: {log_e}")
            # --- End Logging ---
            
            # Clear selection details & update states
            return gallery_data, msg, "Click a showbag image above to see details.", None, "", top_id, new_ids
        
        def update_selected_details(evt: gr.SelectData):
            details, showbag_id = get_selected_showbag_info(evt, df_loaded)
            return details, showbag_id, "" # Clear vote confirm message on new selection

        # Add a function to update the slider label
        def update_slider_label(price):
            return gr.Slider(label=f"Show bags cheaper than ${price:.0f}")

        # Function to update the shortlist display textbox
        def update_shortlist_display(shortlist):
            return "\n".join(shortlist) if shortlist else "Like showbags to add them here!"

        # Connect search triggers
        search_box.submit(fn=update_gallery_wrapper, 
                           inputs=[search_box, price_slider, last_search_ids_state], # Add cached_ids input
                           # Update last_search_ids_state in output
                           outputs=[gallery, status_message, selected_info, selected_showbag_id_state, vote_confirm_message, top_result_id_state, last_search_ids_state]) 
        search_button.click(fn=update_gallery_wrapper, 
                            inputs=[search_box, price_slider, last_search_ids_state],
                            outputs=[gallery, status_message, selected_info, selected_showbag_id_state, vote_confirm_message, top_result_id_state, last_search_ids_state])

        # Connect slider release - Triggered only by price change, so pass EMPTY query
        price_slider.release(fn=update_gallery_wrapper, 
                             # Pass empty query, current price, and cached IDs
                             inputs=[gr.State(""), price_slider, last_search_ids_state], 
                             outputs=[gallery, status_message, selected_info, selected_showbag_id_state, vote_confirm_message, top_result_id_state, last_search_ids_state]) # Update cached IDs (should be unchanged)
        
        # Connect slider change event to update its label
        price_slider.change(fn=update_slider_label, inputs=[price_slider], outputs=[price_slider])

        # Connect gallery select event to update selected details
        gallery.select(fn=update_selected_details, 
                       inputs=None, 
                       outputs=[selected_info, selected_showbag_id_state, vote_confirm_message])
        
        # --- Connect the like/dislike buttons and updated shortlist display ---
        def handle_vote_and_update_display(vote_type, showbag_id, current_shortlist):
            """Handle the vote and update the shortlist display in one function"""
            vote_message, new_shortlist = handle_vote(vote_type, showbag_id, current_shortlist)
            shortlist_text = update_shortlist_display(new_shortlist)
            return vote_message, new_shortlist, shortlist_text

        # Connect vote buttons, passing shortlist_state and shortlist_display
        like_button.click(fn=handle_vote_and_update_display, 
                         inputs=[gr.State('up'), selected_showbag_id_state, shortlist_state], 
                         outputs=[vote_confirm_message, shortlist_state, shortlist_display])
        dislike_button.click(fn=handle_vote_and_update_display, 
                            inputs=[gr.State('down'), selected_showbag_id_state, shortlist_state], 
                            outputs=[vote_confirm_message, shortlist_state, shortlist_display])

        # Initial load - Final corrected wrapper function
        def initial_load_wrapper():
            """Load initial data without relying on state change events"""
            # Call search_and_display directly (no cached IDs initially)
            gallery_data, msg, top_id, initial_ids = search_and_display('', 50, df_loaded, None) 
            # Initialize shortlist display directly
            shortlist_text = update_shortlist_display([])
            initial_selected_info = 'Click a showbag image above to see details.'
            initial_selected_id = None
            initial_vote_confirm = ''
            # Return values matching outputs list (now 8 items)
            return (
                gallery_data, msg, initial_selected_info, initial_selected_id, 
                initial_vote_confirm, top_id, initial_ids, shortlist_text 
            )

        # Connect the app load event to initialize everything
        app.load(fn=initial_load_wrapper, 
                 inputs=None, 
                 outputs=[gallery, status_message, selected_info, selected_showbag_id_state, 
                          vote_confirm_message, top_result_id_state, last_search_ids_state, 
                          shortlist_display])

    return app

# --- Build the UI Instance AT MODULE LEVEL ---
print("app.py: Attempting to build Gradio UI instance...")
gradio_ui = None # Initialize gradio_ui to None
try:
    if showbags_index is not None and showbags_df is not None:
        # Pass the loaded data directly to the build function
        gradio_ui_instance = build_ui(showbags_df, showbags_index)
        # Check if build_ui returned a valid Gradio Blocks object
        if isinstance(gradio_ui_instance, gr.Blocks):
            gradio_ui = gradio_ui_instance
            print(f"app.py: Gradio UI instance created successfully (type: {type(gradio_ui)}).")
        else:
            print(f"app.py: build_ui did not return a Gradio Blocks instance (returned type: {type(gradio_ui_instance)}). Creating error UI.")
            # Fall through to create error UI
    else:
        print("app.py: Data loading failed prior to UI build. Creating error UI.")
        # Fall through to create error UI

    # If gradio_ui is still None (either data failed or build_ui failed), create error UI
    if gradio_ui is None:
        with gr.Blocks() as error_ui_instance:
            gr.Markdown("# Application Startup Error")
            gr.Markdown("Failed to load data or build the main UI during application startup.")
        gradio_ui = error_ui_instance # Assign error UI
        print(f"app.py: Error UI instance created (type: {type(gradio_ui)}).")

except Exception as e:
    print("app.py: CRITICAL ERROR during module-level UI build!")
    import traceback
    traceback.print_exc() # Print detailed traceback
    print("app.py: Creating fallback error UI due to exception.")
    # Ensure gradio_ui is assigned an error block even if exception occurred
    with gr.Blocks() as error_ui_instance:
        gr.Markdown("# Application Startup Exception")
        gr.Markdown(f"An unexpected error occurred during UI initialization: {e}")
    gradio_ui = error_ui_instance
    print(f"app.py: Exception fallback UI instance created (type: {type(gradio_ui)}).")

# --- Main execution block for LOCAL running ONLY ---
if __name__ == "__main__":
    print("Running app.py locally...")
    # The UI is already built at module level. Just launch it.
    if gradio_ui:
        print("Launching Gradio App...")
        # Set share=True for potential external access if needed locally
        gradio_ui.launch(share=False)
    else:
        # This case should technically not happen due to the error UI fallback
        print("Error: Gradio UI instance not found.")

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