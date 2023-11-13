import subprocess
import pkg_resources

def generate_requirements_file():
    '''Generate a requirements.txt file for the project'''
    # List of packages you want to include in the requirements file
    packages_to_include = [
        "scikit-learn", "wandb", "tensorflow", "pandas", "numpy",
        "madgwickahrs", "geopy", "matplotlib", "seaborn", "folium"
    ]

    # Get the list of installed packages using pip freeze
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    # Write the required packages and their versions to a requirements.txt file
    with open('requirements.txt', 'w') as file:
        for package in packages_to_include:
            version = installed_packages.get(package)
            if version:
                file.write(f'{package}=={version}\n')

    print("requirements.txt file has been generated.")

# Call the function to generate the requirements.txt file
generate_requirements_file()
