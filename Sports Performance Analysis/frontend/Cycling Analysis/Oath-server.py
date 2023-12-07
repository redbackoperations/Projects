from flask import Flask, request, redirect
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import WebApplicationClient
import config

app = Flask(__name__)

# Replace with your client ID, client secret, and registered redirect_uri
client_id = config.client_id
client_secret = config.client_secret
redirect_uri = 'http://localhost:5000/callback'

# OAuth endpoints given in the Strava API documentation
authorization_base_url = 'https://www.strava.com/oauth/authorize'
token_url = 'https://www.strava.com/oauth/token'

@app.route("/")
def index():
    """Step 1: User Authorization.
    Redirect the user/resource owner to the OAuth provider (i.e. Strava)
    using an URL with a few key OAuth parameters.
    """
    strava = OAuth2Session(client_id, redirect_uri=redirect_uri)
    authorization_url, state = strava.authorization_url(authorization_base_url)

    # Debugging: Print the full callback URL
    print("Authorization URL:", authorization_url)

    # State is used to prevent CSRF, keep this for later.
    session['oauth_state'] = state
    return redirect(authorization_url)

@app.route("/callback", methods=["GET"])
def callback():
    # Debugging: Print the full callback URL
    print("Callback URL:", request.url)

    """ Step 2: User authorization, this happens on the provider."""
    strava = OAuth2Session(client_id, redirect_uri=redirect_uri)
    
    # Debugging: Print the full callback URL
    print("Callback URL:", request.url)

    # Fetch the token
    token = strava.fetch_token(token_url, client_secret=client_secret,
                               authorization_response=request.url)

    return f"Access token: {token}"

if __name__ == "__main__":
    app.secret_key = 'SECRET_KEY'  # Replace with a real secret key
    app.run(debug=True)
