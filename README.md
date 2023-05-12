Initialize feedback database with this before running apps:

```
import sqlite3

# Connect to the database
conn = sqlite3.connect('llm_quality.db')
c = conn.cursor()


# Commit changes and close the connection
conn.commit()
conn.close()
```

You can then run:

`streamlit run Example_Application.py` - Template app where llm is called and feedback is logged.

`streamlit run Leaderboard.py` - Run the application.
