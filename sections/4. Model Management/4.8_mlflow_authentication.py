from mlflow.server import get_app_client
import os

tracking_uri = "http://localhost:5000"
auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)

# Set username and password
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

# Change password
auth_client.update_user_password(username="admin", password="zG*8!7p@TKmS")
