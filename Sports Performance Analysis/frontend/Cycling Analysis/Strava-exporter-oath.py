from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient
import config

client_id = config.client_id
client_secret = config.client_secret

# Create a session
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Fetch an access token
response = oauth.fetch_token(token_url='https://www.strava.com/oauth/token',
                             client_id=client_id,
                             client_secret=client_secret,
                             include_client_id=True)  # Some APIs require this
print(response)


# Use the token to make authenticated requests
response = oauth.get('https://www.strava.com/api/v3/athlete')

# Print the response
print(response.json())
