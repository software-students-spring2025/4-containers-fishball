import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DBNAME = os.getenv("MONGO_DBNAME")
COLLECTION = "test"

client = MongoClient(MONGO_URI)
db = client[MONGO_DBNAME]
collection = db[COLLECTION]

def test_mongo():
    test_doc = {"test": "MongoDB is working!"}
    
    inserted = collection.insert_one(test_doc)
    print(f"Inserted document ID: {inserted.inserted_id}")
    
    fetched = collection.find_one({"_id": inserted.inserted_id})
    print(f"Fetched document: {fetched}")
    
    collection.delete_one({"_id": inserted.inserted_id})
    print("Test document deleted.")

if __name__ == "__main__":
    test_mongo()