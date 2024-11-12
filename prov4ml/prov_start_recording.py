
import argparse
import subprocess

def main(): 
    subprocess.call(["script out.log"], shell=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a PROV-JSON file to a DOT file')
    args = parser.parse_args()
    
    main()