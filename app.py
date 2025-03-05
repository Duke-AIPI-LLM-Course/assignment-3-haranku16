# Frontend application implementation (Streamlit)
import streamlit as st
from scripts.vectordb import VectorDatabase
from scripts.rag import RetrievalAugmentedGenerator

def initialize_session_state():
    """
    Initialize Streamlit session state variables for chat messages and uploaded files.
    Creates empty lists/sets if they don't exist in the session state.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "files" not in st.session_state:
        st.session_state.files = set()

def display_chat():
    """
    Display the chat history including user messages, assistant responses, and context.
    Each message is displayed in a chat bubble with an expandable context section.
    """
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "context" in message:
                with st.expander("View Source Context"):
                    st.text(message["context"])

def main():
    """
    Main application function that sets up the Streamlit interface.
    Handles file management in the sidebar and implements the chat interface
    with RAG-based document question answering functionality.
    """
    st.set_page_config(layout="wide")
    initialize_session_state()

    # Sidebar for file management
    with st.sidebar:
        st.title("File Management")
        st.write("Upload and manage your documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload text files",
            type="txt",
            accept_multiple_files=True,
            key="file_uploader"
        )

        # Process uploaded files
        with VectorDatabase() as db:
            if uploaded_files:
                for file in uploaded_files:
                    if file.name not in st.session_state.files:
                        content = file.read().decode("utf-8")
                        db.put(file.name, content)
                        st.session_state.files.add(file.name)
                        st.success(f"Added: {file.name}")

            # Display managed files with delete buttons
            if st.session_state.files:
                st.write("Managed Files:")
                for file in sorted(st.session_state.files):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(file)
                    with col2:
                        if st.button("Delete", key=f"del_{file}"):
                            db.delete(file)
                            st.session_state.files.remove(file)
                            st.rerun()

    # Main chat interface
    st.title("Document Q&A")
    st.write("Ask questions about your uploaded documents")

    # Display chat history
    display_chat()

    # Chat input
    if query := st.chat_input("Ask a question about your documents"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Generate response using RAG
        with VectorDatabase() as db:
            rag = RetrievalAugmentedGenerator(db)
            result = rag.generate(query)
            
            # Add assistant response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "context": result["context"]
            })
        
        st.rerun()

if __name__ == "__main__":
    main()
