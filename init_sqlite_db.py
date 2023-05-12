import sqlite3

# Connect to the database
conn = sqlite3.connect('llm_quality.db')
c = conn.cursor()

# Commit changes and close the connection
conn.commit()
conn.close()
