import subprocess
import sys
from rag.generation.build_prompt  import rewrite_query_with_full_history,rewrite_and_classify_query
if __name__ == "__main__":
    # from rag.embedding.embed import embedding
    # embedding()
    #subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_agent.py"])
    