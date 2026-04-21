import subprocess
import sys

if __name__ == "__main__":
    #from rag.embedding.embed import embedding
    #embedding()
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
