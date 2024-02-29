import subprocess
import sys

def get_installed_packages():
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=freeze'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        else:
            print("Error: Unable to get the list of installed packages.")
            return []
    except FileNotFoundError:
        print("Error: 'pip' command not found. Make sure you have pip installed.")
        return []

def create_requirements_txt(packages, output_file='requirements-pip.txt'):
    with open(output_file, 'w') as f:
        for package in packages:
            f.write(package + '\n')

if __name__ == "__main__":
    installed_packages = get_installed_packages()
    if installed_packages:
        create_requirements_txt(installed_packages)
        print("requirements.txt created successfully.")
    else:
        print("No packages found.")
