
import subprocess
import os
import sys

def _get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def _get_git_remote_url():
    try:
        remote_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], stderr=subprocess.DEVNULL).strip().decode()
        return remote_url
    except subprocess.CalledProcessError:
        print("> get_git_remote_url() Repository not found")
        return None  # No remote found
    
def _requirements_lookup(path): 
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename == "requirements.txt": 
                return os.path.join(root, filename)
    return None
    
def _get_source_files(): 
    PROJECT_ROOT = os.path.dirname(os.path.relpath(sys.argv[0]))

    source_files = [
        os.path.relpath(m.__file__)
        for m in sys.modules.values()
        if hasattr(m, "__file__")
        and m.__file__
        and m.__file__.endswith(".py")
        and os.path.relpath(m.__file__).startswith(PROJECT_ROOT)
    ]

    return set(source_files)

