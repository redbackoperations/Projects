=================================================
Environment setup:

# If venv is existing within the backend foders please delete it and create a new one using the following commad


python3 -m venv .venv

Activate virtual environment:

On windows CMD
.venv\Scripts\activate

On Unix or MacOS
source .venv/bin/activate

Install necessary packages: 
pip install Django
pip install djangorestframework

=================================================
# How to use 
Execute the following commands:

cd "Project 8 - Backend"

venv\Scripts\activate

python manage.py runserver

# once the server is running go ahead and hold Ctrl and click on the link.

go to urls.by to see availabe likns that can be use. Example http://127.0.0.1:8000/login/

To manage the backend database visit http://127.0.0.1:8000/admin

please ask your supervisor for admin credintials.