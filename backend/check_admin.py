import sqlite3

db = "data/burnui.sqlite3"

conn = sqlite3.connect(db)

print("USERS:")
for row in conn.execute("SELECT id, email, role FROM users"):
    print(row)

conn.close()