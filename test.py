from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os

# MongoDB connection URI
# uri = "mongodb+srv://utkarshraj7217:utkarshraj7217@cluster0.e6wue.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
uri = os.getenv("MONGODB_URL")
# uri = "mongodb+srv://utkarshraj7217:<db_password>@cluster0.e6wue.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

try:
    # Connect to MongoDB
    client = MongoClient(uri)
    # Test the connection
    client.admin.command('ping')
    print("Connected to MongoDB server successfully.")

    # List all database names
    database_names = client.list_database_names()
    print("Databases:", database_names)

except ConnectionFailure as cf:
    print("Connection to MongoDB server failed:", cf)
except Exception as e:
    print("An error occurred:", e)


