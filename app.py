# Frontend application implementation (Streamlit)
import streamlit as st
from scripts.vectordb import VectorDatabase

def main():
    st.title("Vector Database Search")
    st.write("Search for documents in the vector database")

    # Initialize vector database
    with VectorDatabase() as db:
        # Take a file upload of a text file and store it in the vector database
        files = st.file_uploader("Upload a text file", type="txt", accept_multiple_files=True)
        if files:
            for file in files:
                db.put(file.name, file.read().decode("utf-8"))

        # Search for documents
        query = st.text_input("Enter a search query")
        if st.button("Search"):
            st.write(db.search(query))

if __name__ == "__main__":
    main()