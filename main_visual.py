import subprocess
import sys

def main():
    print("Iniciando la aplicación web de Machine Learning...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/streamlit_app.py"])

if __name__ == "__main__":
    main()