import os
import subprocess

def run_script(script_name):
    print(f"\n{'='*50}\nRunning {script_name}...\n{'='*50}")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
        exit(1)

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    scripts = [
        "src/download_data.py",
        "src/clean.py",
        "src/create_db.py",
        "src/analysis.py",
        "src/train_model.py",
    ]

    for script in scripts:
        run_script(script)

    print("\nPipeline completed successfully. You can now run:")
    print("  uvicorn app.main:app --reload")
    print("  streamlit run frontend/dashboard.py")

if __name__ == "__main__":
    main()
