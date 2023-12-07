import requests
import config

token = config.token

# The Strava API URL for athlete activities
url = 'https://www.strava.com/api/v3/athlete/activities'
#url = 'https://www.strava.com/api/v3/athlete'

# NOTE: This code works for the athlete endpoint but not the activities endpoint. It is 
# likely that the activities endpoint requires a different scope and therefore a different
# token. The scope for the activities endpoint is 'activity:read_all' (see 
# https://developers.strava.com/docs/authentication/#request-access for more information).

# Parameters for the request (if you have specific values for before, after, page, and per_page, add them here)
params = {
#    'before': '',  # Add your 'before' parameter here if needed
#    'after': '',   # Add your 'after' parameter here if needed
#    'page': '',    # Add your 'page' parameter here if needed
#    'per_page': '' # Add your 'per_page' parameter here if needed
    #'scope': 'activity:read_all' # Adding a scope here doesn't work.
}

# Headers including the authorization token
headers = {
    'Authorization': f'Bearer {token}'
}

# Make the GET request
response = requests.get(url, headers=headers, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Print the response content (or handle it as needed)
    print(response.json())
else:
    # Handle errors (e.g., print error message)
    print(f'Error: {response.status_code}')
    print(f'Response: {response.content}')
