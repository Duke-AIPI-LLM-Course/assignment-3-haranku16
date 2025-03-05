[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/a87xfYGP)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=18275281)

### Workspace setup

```
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate # Or on Windows, .venv/Scripts/activate

# Install requirements
pip install -r requirements.txt
```

### Running evaluation
The evaluation script also pulls the book from Project Gutenberg, which can be useful for using in the Streamlit application.

```
python evaluate.py
```

This produces a file called `evaluation_results.py`.

### Running the Streamlit app

```
streamlit run app.py
```

### Presentation & Demo

You can find the presentation video [here](https://youtu.be/GJIQqk1Se1Q).

### Deployed Streamlit application

https://duke-aipi-llm-course-assignment3-haranku16.streamlit.app/

### Other useful links

[H.G. Wells' The Time Machine, on Project Gutenberg](https://www.gutenberg.org/ebooks/35)