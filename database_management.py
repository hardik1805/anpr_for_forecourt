import pymongo

# MongoDB connection parameters
mongo_client = pymongo.MongoClient("mongodb+srv://hardik:1182@cluster0.wdbyzcz.mongodb.net/")
print('MongoDB connected successfully!!')
db = mongo_client["licence_plate_management"]  # Replace with your database name
collection = db["blocked_plates"]  # Replace with your collection name


def check_blocked_list(license_number):
    # Search for documents with the given registration number
    query = {"registrationnumber": license_number.upper()}
    result = collection.find_one(query)

    return result


# print(check_blocked_list("GU56OHB"))
# print(check_blocked_list("gu56OHB"))


# Close the MongoDB client when done
# mongo_client.close()
