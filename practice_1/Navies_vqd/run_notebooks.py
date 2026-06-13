import json
import os
import sys
import ssl

# Bypass SSL verification for nltk downloads (common issue on macOS)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Configure matplotlib to use a non-interactive backend 'Agg'
# to prevent plt.show() from opening GUI windows and blocking execution.
try:
    import matplotlib
    matplotlib.use('Agg')
    print("Matplotlib backend successfully set to 'Agg'.")
except Exception as e:
    print(f"Warning: Could not set matplotlib backend: {e}")

# Determine workspace_root dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir points to practice_1/Navies_vqd, so workspace_root is parent of practice_1
workspace_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

def run_notebook(nb_path):
    print(f"\n========================================\nRunning notebook: {nb_path}\n========================================")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    globals_dict = {
        '__file__': nb_path,
        '__name__': '__main__'
    }
    
    nb_dir = os.path.dirname(nb_path)
    old_cwd = os.getcwd()
    os.chdir(nb_dir)
    sys.path.insert(0, nb_dir)
    
    try:
        cell_idx = 0
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                cell_idx += 1
                source = "".join(cell['source'])
                if not source.strip():
                    continue
                print(f"Executing Code Cell {cell_idx} (id: {cell.get('id', 'N/A')})...")
                try:
                    exec(source, globals_dict)
                    
                    # Register NaiveBayesClassifierFromScratch and TFIDFVectorizerFromScratch dynamically
                    if 'NaiveBayesClassifierFromScratch' in globals_dict:
                        sys.modules['__main__'].__dict__['NaiveBayesClassifierFromScratch'] = globals_dict['NaiveBayesClassifierFromScratch']
                    if 'TFIDFVectorizerFromScratch' in globals_dict:
                        sys.modules['__main__'].__dict__['TFIDFVectorizerFromScratch'] = globals_dict['TFIDFVectorizerFromScratch']
                except Exception as e:
                    print(f"ERROR in Cell {cell_idx} (id: {cell.get('id', 'N/A')}):\n{e}")
                    import traceback
                    traceback.print_exc()
                    raise e
        print(f"Notebook {os.path.basename(nb_path)} ran successfully!")
    finally:
        os.chdir(old_cwd)
        if nb_dir in sys.path:
            sys.path.remove(nb_dir)

if __name__ == "__main__":
    p1_solution_dir = os.path.join(workspace_root, "practice_1", "Navies_vqd")
    notebooks = [
        "02_data_checks.ipynb",
        "03_data_cleaning.ipynb",
        "04_eda.ipynb",
        "05_text_preprocessing.ipynb",
        "06_model_training.ipynb",
        "07_evaluation.ipynb",
        "09_compare_preprocessing.ipynb"
    ]
    
    try:
        for nb in notebooks:
            run_notebook(os.path.join(p1_solution_dir, nb))
        print("\nAll notebooks (02-07, 09) executed successfully and validated!")
    except Exception as e:
        print(f"\nExecution failed with error: {e}")
        sys.exit(1)
