# tokengenerator.py (for Plaid OpenAPI SDK)
from plaid.api import plaid_api
from plaid.model.sandbox_public_token_create_request import SandboxPublicTokenCreateRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.products import Products
from plaid import Configuration, ApiClient, Environment
import os
from dotenv import load_dotenv

load_dotenv()
# Load config from .env
PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
PLAID_SECRET = os.getenv("PLAID_SECRET")
PLAID_ENV = os.getenv("PLAID_ENV", "sandbox")


# Configure Plaid client
configuration = Configuration(
    host=getattr(Environment, PLAID_ENV.capitalize()),
    api_key={
        "clientId": PLAID_CLIENT_ID,
        "secret": PLAID_SECRET,
    }
)
api_client = ApiClient(configuration)
client = plaid_api.PlaidApi(api_client)

# Step 1: Create sandbox public_token
request = SandboxPublicTokenCreateRequest(
    institution_id="ins_109508",
    initial_products=[Products("transactions")],
)

response = client.sandbox_public_token_create(request)
public_token = response["public_token"]

# Step 2: Exchange for access_token
exchange_request = ItemPublicTokenExchangeRequest(public_token=public_token)
exchange_response = client.item_public_token_exchange(exchange_request)

access_token = exchange_response["access_token"]

print("✅ Sandbox Access Token:", access_token)
