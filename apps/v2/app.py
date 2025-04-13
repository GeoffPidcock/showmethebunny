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
import sqlite3  # Add sqlite3 import
from datetime import datetime  # Add datetime import
import uuid  # Add uuid import

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

# Database setup
DB_FILE = "showbag_logs.db"

def init_db():
    """Initialize the SQLite database and create tables if they don't exist."""
    try:
        conn = sqlite3.connect(DB_FILE)
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
        print(f"Database initialized successfully at {DB_FILE}")
    except Exception as e:
        print(f"Error initializing database: {e}")

init_db() # Initialize the database when the app starts

# Initialize OpenAI embedding model with API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embed_model = OpenAIEmbedding()
Settings.embed_model = embed_model

# Path to the showbags CSV file
SHOWBAGS_PATH = Path("../../data/showbags/showbags_enriched.csv")

def load_showbags_data():
    """Load showbags DataFrame and create LlamaIndex."""
    try:
        # Check if the CSV file exists
        if not SHOWBAGS_PATH.exists():
            raise FileNotFoundError(f"Showbags data file not found at {SHOWBAGS_PATH}")
            
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(SHOWBAGS_PATH)

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

def log_interaction(user_query: str, retrieved_context: str, final_response: str):
    """Log interaction details to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        c.execute("INSERT INTO interactions (id, timestamp, user_query, retrieved_context, final_response) VALUES (?, ?, ?, ?, ?)",
                  (interaction_id, timestamp, user_query, retrieved_context, final_response))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging interaction: {e}")

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
        # Get the pre-loaded index and dataframe
        index = app_data["index"] # Use app_data
        if not index:
            return "Error: Could not load showbags index. Please check logs."
        
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

# Reference to hold the index and dataframe globally
app_data = {"index": None, "df": None}

# Load data when the script starts
app_data["index"], app_data["df"] = load_showbags_data()

# Check if data loading was successful before proceeding
if app_data["df"] is None:
    print("Failed to load showbag data. Exiting.")
    # Optionally raise an exception or exit
    # raise SystemExit("Failed to load data.") 

# Keep the functions related to the NEW UI
def log_vote(showbag_id: str, vote_type: str):
    """Log a vote action to the SQLite database."""
    if not showbag_id:
        print("Error logging vote: No showbag ID provided.")
        return
    try:
        conn = sqlite3.connect(DB_FILE)
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

def search_and_display(search_query: str, max_price: float, df: pd.DataFrame, cached_ids: List[str] | None = None):
    """Filters DataFrame based on semantic search (if query provided), price, and optional cached IDs."""
    if df is None: return [], "Error: Showbag data not loaded.", None, None # gallery_data, msg, top_id, new_ids
    if app_data["index"] is None: return [], "Error: Showbag index not loaded.", None, None

    SIMILARITY_THRESHOLD = 0.75 
    df['price_numeric'] = pd.to_numeric(df['price_numeric'], errors='coerce')
    df['id'] = df['id'].fillna('').astype(str)

    working_df = df.copy()
    search_performed = False
    top_result_id = None
    current_ordered_ids = None # Holds IDs used for filtering, returned at end

    # --- Semantic Search (only if search_query is provided) ---
    if search_query and search_query.strip():
        search_performed = True
        try:
            retriever = VectorIndexRetriever(index=app_data["index"], similarity_top_k=50)
            retrieved_nodes_with_scores = retriever.retrieve(search_query)
            high_similarity_nodes = [n for n in retrieved_nodes_with_scores if n.score >= SIMILARITY_THRESHOLD]
            # Get NEW ordered IDs from this search
            current_ordered_ids = list(dict.fromkeys([str(node.metadata['id']) for node in high_similarity_nodes if 'id' in node.metadata]))
        except Exception as e:
            print(f"Error during semantic search: {e}")
            # On error, maybe revert to full list or return error? Clear IDs for now.
            current_ordered_ids = None 
            return [], f"Error during search: {e}", None, current_ordered_ids
    else:
        # No search query provided, use cached IDs if available
        current_ordered_ids = cached_ids
    
    # --- Filtering based on IDs (either new search or cached) ---
    if current_ordered_ids: 
        search_performed = True # Mark as filtered if IDs were used
        working_df = working_df[working_df['id'].isin(current_ordered_ids)]
        # Reapply ordering based on IDs used (new or cached)
        if not working_df.empty:
             working_df['id'] = pd.Categorical(working_df['id'], categories=current_ordered_ids, ordered=True)
             working_df = working_df.sort_values('id')
    # else: No query and no cached IDs, proceed with full dataframe

    # --- Price Filter ---
    final_filtered_df = working_df[working_df['price_numeric'] <= max_price].copy()

    # Get top result ID after price filtering
    if not final_filtered_df.empty:
        top_result_id = final_filtered_df.iloc[0]['id']
    
    # --- Format for Gallery & Message ---
    gallery_data = []
    for _, row in final_filtered_df.iterrows(): 
        caption = f"{row.get('name', 'N/A')} - ${row.get('price_numeric', 'N/A'):.2f}"
        image_url = row.get('image_url', None)
        gallery_data.append((image_url, caption))

    # --- Generate Status Message --- 
    num_results = len(gallery_data)
    if num_results > 0:
        if search_performed:
             query_part = f" matching '{search_query}'" if (search_query and search_query.strip()) else " from last search" 
             message = f"Found {num_results} showbags{query_part} (and price filter)."
        else:
             message = f"Showing {num_results} showbags based on the price filter."
    else:
        if search_performed:
            query_part = f" matching '{search_query}'" if (search_query and search_query.strip()) else " from last search" 
            message = f"No showbags found{query_part} and the price filter."
        else:
            message = "No showbags found within the selected price range."

    # Return gallery data, message, top_id, and the IDs used for filtering (new or cached or None)
    return gallery_data, message, top_result_id, current_ordered_ids

def get_selected_showbag_info(evt: gr.SelectData, df: pd.DataFrame):
    """Get the ID and display info of the selected showbag based on the gallery selection event."""
    if df is None or evt is None:
        return "Select a showbag to see details.", None

    try:
        selected_index = evt.index
        
        # evt.value is a dictionary like: {'image': {...}, 'caption': 'Name - $Price'}
        # Extract the caption string from the dictionary
        selected_caption = evt.value.get('caption') # Use .get for safety
        if not selected_caption or not isinstance(selected_caption, str):
             print(f"Error: Could not extract caption string from evt.value: {evt.value}")
             return "Error extracting caption from selected item.", None
        
        # Now split the caption string
        name_part = selected_caption.split(' - $')[0]
        
        # Find the corresponding row in the original DataFrame
        # Use boolean indexing which is safer than relying on iloc[0] if duplicates exist
        matching_rows = df[df['name'] == name_part]
        if matching_rows.empty:
            print(f"Error: Could not find showbag with name: {name_part}")
            return f"Could not find details for {name_part}", None
            
        selected_row = matching_rows.iloc[0] # Take the first match
        
        showbag_id = selected_row.get('id') # Get the actual ID from the DataFrame
        if not showbag_id:
             # Fallback if ID is missing in the DataFrame for some reason
             showbag_id = str(uuid.uuid4()) 
             print(f"Warning: Using generated UUID for showbag {name_part} as ID was missing.")

        # Ensure required fields exist, provide defaults
        name = selected_row.get('name', 'N/A')
        price = selected_row.get('price_numeric', 'N/A')
        retail_value = selected_row.get('retail_value_numeric', 'N/A')
        stands = selected_row.get('stand_numbers', 'N/A')
        # img_desc = selected_row.get('img_description', 'N/A') # No longer displayed
        included_items = selected_row.get('included_items', 'N/A')

        # Format price and retail value safely
        try:
            price_str = f"${price:.2f}" if pd.notna(price) else "N/A"
        except TypeError:
            price_str = "N/A"
        try:
            retail_value_str = f"${retail_value:.2f}" if pd.notna(retail_value) else "N/A"
        except TypeError:
            retail_value_str = "N/A"

        details = f"**{name}**\n"
        details += f"Price: {price_str}\n"
        details += f"Retail Value: {retail_value_str}\n"
        details += f"Stands: {stands}\n\n"
        details += f"**Includes:**\n{included_items}"
        
        return details, showbag_id
    except Exception as e:
        # Print the actual exception for debugging
        print(f"Error getting selected info: {e}, Event Data: {evt}") 
        # Avoid showing the raw dictionary in the UI error message
        error_val_display = evt.value.get('caption', 'selected item') if isinstance(evt.value, dict) else str(evt.value)
        return f"Error displaying details for: {error_val_display}", None

# Define handle_vote OUTSIDE build_ui - now handles shortlist
def handle_vote(vote_type, showbag_id, current_shortlist):
    if not showbag_id:
        return "Please select a showbag before voting.", current_shortlist # Return unchanged shortlist
    
    log_vote(showbag_id, vote_type)
    
    # Initialize shortlist if None
    if current_shortlist is None:
        current_shortlist = []

    new_shortlist = list(current_shortlist) # Make a mutable copy
    message = f"Vote ({vote_type}) recorded." # Base message

    if vote_type == 'up':
        # Add to shortlist if liked and not already present
        # Find the showbag details from the main DataFrame
        df = app_data.get("df")
        if df is not None:
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
                
                shortlist_item_str = f"- {name} ({price_str}) - {stands}"
                
                # Check if already in shortlist (based on name for simplicity)
                already_exists = any(name in item for item in new_shortlist)
                
                if not already_exists:
                    new_shortlist.append(shortlist_item_str)
                    message += f" Added '{name}' to shortlist!"
                else:
                    message += f" '{name}' is already in the shortlist."
            else:
                message += " Could not find showbag details to add to shortlist."
        else:
            message += " Error accessing showbag data for shortlist."
    elif vote_type == 'down':
        # Optionally remove from shortlist if disliked?
        # For now, just log the downvote.
        pass

    return message, new_shortlist

# Keep the NEW UI build function
def build_ui(df, index):
    if df is None:
        gr.Markdown("# Error: Showbag data could not be loaded. Check logs.")
        return None

    # Determine price range for slider
    df['price_numeric'] = pd.to_numeric(df['price_numeric'], errors='coerce')
    min_price = 0
    max_price = df['price_numeric'].dropna().max()
    if pd.isna(max_price):
        max_price = 50 # Default max price if none found
    max_price = max(min_price + 1, max_price) # Ensure max > min

    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# Find Your Awesome Easter Showbag! 🎒✨")
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
                price_slider = gr.Slider(minimum=min_price, maximum=max_price, value=max_price, step=1, label=f"Show bags cheaper than ${max_price}")
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
            gallery_data, msg, top_id, new_ids = search_and_display(query, price, df, cached_ids)
            # Clear selection details & update states
            return gallery_data, msg, "Click a showbag image above to see details.", None, "", top_id, new_ids
        
        def update_selected_details(evt: gr.SelectData):
            details, showbag_id = get_selected_showbag_info(evt, df)
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
        
        # Connect vote buttons - pass shortlist state IN and OUT
        like_button.click(fn=handle_vote, 
                          inputs=[gr.State('up'), selected_showbag_id_state, shortlist_state], 
                          outputs=[vote_confirm_message, shortlist_state])
        dislike_button.click(fn=handle_vote, 
                             inputs=[gr.State('down'), selected_showbag_id_state, shortlist_state], 
                             outputs=[vote_confirm_message, shortlist_state]) # Downvote currently doesn't modify list, but wiring is symmetrical

        # Connect shortlist_state changes to update the display textbox
        shortlist_state.change(fn=update_shortlist_display, 
                               inputs=[shortlist_state], 
                               outputs=[shortlist_display])

        # Initial load - Final corrected wrapper function
        def initial_load_wrapper():
            # Call search_and_display directly (no cached IDs initially)
            gallery_data, msg, top_id, initial_ids = search_and_display('', max_price, df, None) 
            shortlist_text = update_shortlist_display([])
            initial_selected_info = 'Click a showbag image above to see details.'
            initial_selected_id = None
            initial_vote_confirm = ''
            # Return values matching outputs list (now 8 items)
            return (
                gallery_data, msg, initial_selected_info, initial_selected_id, 
                initial_vote_confirm, top_id, initial_ids, shortlist_text 
            )

        app.load(fn=initial_load_wrapper, 
                 inputs=None, 
                 # Add last_search_ids_state to outputs
                 outputs=[gallery, status_message, selected_info, selected_showbag_id_state, vote_confirm_message, top_result_id_state, last_search_ids_state, shortlist_display]) 

    return app

# Check if data loading was successful before proceeding
if app_data["df"] is None:
    print("Failed to load showbag data. Cannot build UI.")

# --- Launch the App --- 
if __name__ == "__main__":
    if app_data["df"] is not None:
        # Build the NEW UI using the build_ui function
        app_instance = build_ui(app_data["df"], app_data["index"])
        if app_instance:
            print("Launching Gradio App...")
            app_instance.launch()
        else:
            print("Failed to build Gradio UI.")
    else:
        print("Application cannot start because data failed to load.")

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