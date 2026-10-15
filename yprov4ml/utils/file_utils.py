
import os
import sys
import importlib.metadata
import configparser
from pathlib import Path

def _find_git_dir(start_path="."):
    """Traverses upward to find the .git directory."""
    current = Path(start_path).resolve()
    for parent in [current] + list(current.parents):
        git_dir = parent / ".git"
        if git_dir.is_dir():
            return git_dir
    return None

def _get_git_revision_hash() -> str:
    """Gets the current git commit hash without calling the git CLI."""
    try:
        git_dir = _find_git_dir()
        if not git_dir:
            return None
        
        head_path = git_dir / "HEAD"
        if not head_path.exists():
            return None
        
        head_content = head_path.read_text(encoding="utf-8").strip()
        
        # If it's a symbolic reference (e.g., ref: refs/heads/main)
        if head_content.startswith("ref: "):
            ref_path = git_dir / head_content[5:]
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            
            # Fallback: check packed-refs if the loose ref doesn't exist
            packed_refs_path = git_dir / "packed-refs"
            if packed_refs_path.exists():
                ref_name = head_content[5:]
                for line in packed_refs_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split(" ")
                    if len(parts) == 2 and parts[1] == ref_name:
                        return parts[0]
        else:
            # Detached HEAD state (contains the raw hash directly)
            return head_content
            
    except Exception:
        return None
    return None

def _get_git_remote_url() -> str:
    """Gets the origin remote URL by parsing the local .git/config file."""
    git_dir = _find_git_dir()
    if not git_dir:
        return None
    
    config_path = git_dir / "config"
    if not config_path.exists():
        return None
    
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    
    section = 'remote "origin"'
    if config.has_section(section) and config.has_option(section, 'url'):
        return config.get(section, 'url')
            
    return None


def _requirements_lookup(path): 
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename == "requirements.txt": 
                return os.path.join(root, filename)
    return None

def _generate_requirements(output_dir):
    """
    Generates a requirements.txt file in the specified directory 
    using importlib.metadata from the active Python environment.
    """
    output_path = os.path.join(output_dir, "requirements.txt")
    
    try:
        # Collect all installed distributions and format them like pip freeze
        packages = sorted(
            [f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions()]
        )
        
        # Write the packages to requirements.txt
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(packages) + "\n")
            
        return output_path
        
    except Exception as e:
        print(f"Error generating requirements.txt: {e}", file=sys.stderr)
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
