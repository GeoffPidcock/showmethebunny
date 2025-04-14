from pathlib import Path
import os
import sys
import asyncio
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI
from gradio.routes import mount_gradio_app
import modal

# --- Configuration ---
APP_NAME = "show-me-the-bunny-v2"
CSV_FILENAME = "showbags_enriched.csv"
DB_FILENAME = "showbag_logs.db"
LOCAL_APP_PATH = Path(__file__).parent / "app.py"
LOCAL_DATA_FILE_PATH = Path(__file__).parent.parent.parent / "data" / "showbags" / CSV_FILENAME
REMOTE_DATA_DIR = Path("/data")
REMOTE_CSV_PATH = REMOTE_DATA_DIR / CSV_FILENAME
REMOTE_DB_DIR = Path("/db")
REMOTE_DB_PATH = REMOTE_DB_DIR / DB_FILENAME # Full remote path for DB
LOCAL_TEMP_DB_PATH = Path(".") / DB_FILENAME # Path for DB inside container runtime

# Print paths for debugging
print(f"Local app path: {LOCAL_APP_PATH}")
print(f"Local data file path: {LOCAL_DATA_FILE_PATH}")
print(f"Remote CSV path: {REMOTE_CSV_PATH}")
print(f"Remote DB path: {REMOTE_DB_PATH}")

# Create directory functions for image setup
def create_data_dir():
    """Create the data directory in the container."""
    os.makedirs("/data", exist_ok=True)
    print("Created /data directory")
    
def create_db_dir():
    """Create the database directory in the container."""
    os.makedirs("/db", exist_ok=True)
    print("Created /db directory")

# --- Define Modal Image ---
# Create a lightweight Modal image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gradio==4.26.0",
        "pandas==2.1.4",
        "python-dotenv==1.0.1",
        # Pin pydantic and related packages to compatible versions
        "pydantic==2.10.6",  # Fixed version to avoid 'bool' is not iterable error
        "fastapi==0.110.0",  # Specific FastAPI version compatible with pydantic 2.10.x
        "starlette==0.36.3",  # Specific Starlette version compatible with FastAPI 0.110.0
        # Specify exact versions for compatibility
        "llama-index-core>=0.10.0,<0.13.0",
        "llama-index-embeddings-openai>=0.1.0,<0.4.0",
        "llama-index-llms-openai>=0.1.0,<0.4.0",
        "openai>=0.28.1,<2.0.0",
        "pymupdf",
    )
    # Enable stub volumes for the image using named functions
    .run_function(create_data_dir)
    .run_function(create_db_dir)
    # Copy files
    .copy_local_file(LOCAL_APP_PATH, "/app/app.py")
    .copy_local_file(LOCAL_DATA_FILE_PATH, REMOTE_CSV_PATH)
)

# --- Define Modal App ---
# Create an app with the specified name and image
app = modal.App(APP_NAME, image=image)

# --- Persistent storage for the SQLite database ---
db_volume = modal.Volume.from_name("showbag-db-volume", create_if_missing=True)


@app.function(
    secrets=[modal.Secret.from_name("openai-secret-3")],
    concurrency_limit=1,  # Only one instance to avoid DB conflicts
    allow_concurrent_inputs=1000,  # But handle many concurrent requests
    container_idle_timeout=600,  # Extend timeout to 10 minutes
    timeout=600,  # Set function timeout to 10 minutes
    volumes={REMOTE_DB_DIR: db_volume},  # Mount our DB volume
)
@modal.asgi_app()  # Expose as an ASGI web app
def serve() -> FastAPI:
    """
    Serves the pre-built Gradio UI using FastAPI and handles DB persistence.
    """
    print("Serve function starting...")
    
    # Add to path so imports work correctly
    sys.path.insert(0, "/app")
    
    # --- Create necessary directories ---
    os.makedirs(REMOTE_DATA_DIR, exist_ok=True)
    os.makedirs(REMOTE_DB_DIR, exist_ok=True)
    
    # --- Setup Environment Variables BEFORE importing app ---
    # This is critical because app.py uses these variables at module level
    os.environ["DB_PATH"] = str(LOCAL_TEMP_DB_PATH)
    os.environ["CSV_PATH"] = str(REMOTE_CSV_PATH)
    # Make sure OpenAI API key is accessible from environment
    if "OPENAI_API_KEY" in os.environ:
        print("OPENAI_API_KEY is already set in environment")
    else:
        print("Warning: OPENAI_API_KEY not found in environment")
    
    print(f"Runtime ENV set: DB_PATH={os.getenv('DB_PATH')}, CSV_PATH={os.getenv('CSV_PATH')}")
    
    # --- Copy existing DB from volume before importing app ---
    try:
        # This ensures DB is available during app.py module level execution
        if REMOTE_DB_PATH.exists():
            print(f"Copying initial DB from {REMOTE_DB_PATH} to {LOCAL_TEMP_DB_PATH}")
            LOCAL_TEMP_DB_PATH.write_bytes(REMOTE_DB_PATH.read_bytes())
            print("DB copy successful")
        else:
            print(f"No existing DB found at {REMOTE_DB_PATH}, a new one will be created.")
    except Exception as e:
        print(f"Error copying DB: {e}")
        traceback.print_exc()
        
    # Check if CSV file exists
    if not REMOTE_CSV_PATH.exists():
        print(f"ERROR: CSV file not found at {REMOTE_CSV_PATH}!")
    else:
        print(f"CSV file found at {REMOTE_CSV_PATH}, size: {REMOTE_CSV_PATH.stat().st_size} bytes")
        
    # --- Import app module AFTER environment is configured ---
    try:
        print("Importing app module...")
        print(f"Current sys.path: {sys.path}")
        print(f"Files in /app: {os.listdir('/app')}")
        import app as gradio_app_module
        print("App module imported successfully")
    except Exception as e:
        print(f"ERROR importing app module: {e}")
        traceback.print_exc()
        # Create a fallback Gradio UI with the error message
        import gradio as gr
        with gr.Blocks() as error_ui:
            gr.Markdown("# Application Import Error")
            gr.Markdown(f"Failed to import app module: {str(e)}")
            gr.Markdown("## Traceback")
            gr.Markdown(f"```\n{traceback.format_exc()}\n```")
        gradio_app_module = type('ModuleStub', (), {'gradio_ui': error_ui})()

    # --- DB Persistence Logic ---
    def persist():
        """Copies DB from local runtime back to persistent volume."""
        runtime_db_path_str = os.getenv("DB_PATH", str(LOCAL_TEMP_DB_PATH))
        runtime_db_path = Path(runtime_db_path_str)
        if runtime_db_path.exists():
            print(f"Persisting DB from {runtime_db_path} to {REMOTE_DB_PATH}")
            try:
                REMOTE_DB_PATH.write_bytes(runtime_db_path.read_bytes())
                db_volume.commit()  # Commit changes to the volume
                print("DB persisted successfully")
            except Exception as e:
                print(f"Error persisting DB: {e}")
                traceback.print_exc()
        else:
            print(f"Warning: Local DB file {runtime_db_path} not found for persistence.")

    async def persist_background():
        """Periodically saves the DB in the background."""
        while True:
            try:
                await asyncio.sleep(60)  # Persist every 60 seconds
                persist()
            except Exception as e:
                print(f"Error in background persistence task: {e}")
                traceback.print_exc()
                await asyncio.sleep(10)  # Short delay before retrying

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        """FastAPI lifespan manager for background persistence."""
        print("Lifespan startup: Starting background DB task.")
        task = asyncio.create_task(persist_background())
        try:
            yield  # API runs here
        except Exception as e:
            print(f"Error during API execution: {e}")
            traceback.print_exc()
        finally:
            print("Lifespan shutdown: Persisting DB one last time.")
            try:
                persist()  # Persist on shutdown
            except Exception as e:
                print(f"Error during final DB persist: {e}")
            
            task.cancel()  # Clean up background task
            try:
                await task
            except asyncio.CancelledError:
                print("Background persistence task cancelled.")
            except Exception as e:
                print(f"Error cancelling background task: {e}")

    # Create FastAPI app with lifespan manager
    api = FastAPI(lifespan=lifespan)

    # --- Get the Pre-built UI from app.py ---
    print("Accessing pre-built Gradio UI from gradio_app_module...")
    try:
        blocks = gradio_app_module.gradio_ui
        print(f"Gradio UI accessed successfully, type: {type(blocks)}")
        
        if blocks is None:
            print("CRITICAL ERROR: Imported Gradio UI ('gradio_ui') is None. Creating fallback UI.")
            # Fallback: Create and mount a simple error message app
            import gradio as gr
            with gr.Blocks() as error_ui:
                gr.Markdown("# Application Mount Error")
                gr.Markdown("The main Gradio application UI could not be loaded or built correctly. It returned None.")
            blocks = error_ui
    except Exception as e:
        print(f"Error accessing gradio_ui from module: {e}")
        traceback.print_exc()
        # Create a fallback UI
        import gradio as gr
        with gr.Blocks() as error_ui:
            gr.Markdown("# Application Error")
            gr.Markdown(f"Error accessing Gradio UI: {str(e)}")
            gr.Markdown("## Traceback")
            gr.Markdown(f"```\n{traceback.format_exc()}\n```")
        blocks = error_ui

    # Mount the Gradio app with proper error handling
    try:
        print(f"Mounting Gradio UI to FastAPI...")
        result = mount_gradio_app(app=api, blocks=blocks, path="/")
        print("Gradio UI mounted successfully")
        return result
    except Exception as e:
        print(f"Error mounting Gradio app: {e}")
        traceback.print_exc()
        # Return a simple FastAPI app as fallback
        @api.get("/")
        def root():
            return {
                "error": "Failed to mount Gradio app",
                "details": str(e)
            }
        return api


@app.local_entrypoint()
def main():
    """Local development entry point."""
    print(f"Modal app '{APP_NAME}' is defined.")
    print("To deploy, run: modal deploy apps/v2/modal_wrapper.py")
    print("For local testing, run: python apps/v2/app.py") 