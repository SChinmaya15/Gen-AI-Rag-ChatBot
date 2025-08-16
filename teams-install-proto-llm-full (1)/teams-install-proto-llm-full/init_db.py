import os
from db import init_db
os.makedirs("data", exist_ok=True)
if __name__ == "__main__":
    init_db()
    print("✅ Database initialized")
