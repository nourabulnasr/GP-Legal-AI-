import sqlite3

DB_PATH = r"C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main\legalai.db"
EMAIL = "admin@test.com"

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

cur.execute("SELECT id, email, role FROM users WHERE email = ?", (EMAIL,))
before = cur.fetchone()
print("BEFORE:", before)

if before is None:
    print("\nERROR: admin@test.com not found. Register it first using /auth/register.\n")
else:
    cur.execute("UPDATE users SET role = 'admin' WHERE email = ?", (EMAIL,))
    con.commit()
    cur.execute("SELECT id, email, role FROM users WHERE email = ?", (EMAIL,))
    after = cur.fetchone()
    print("AFTER:", after)

con.close()
