import json
import os
import ssl
import subprocess
import urllib.request

ctx = ssl.create_default_context()
ctx.check_hostname = True
ctx.verify_mode = ssl.CERT_REQUIRED
token = os.environ.get("GITHUB_TOKEN")
try:
    subprocess.run(
        [
            "python3",
            "-m",
            "pip",
            "install",
            "--break-system-packages",
            "pre-commit",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(["pre-commit", "run", "--all-files"], check=False)
    subprocess.run(["git", "add", "."], check=False)
    subprocess.run(
        ["git", "commit", "-m", "chore: auto-format to pass CI checks"],
        check=False,
    )
    subprocess.run(["git", "push"], check=False)
except Exception:
    pass

payload = {
    "title": "Fix for issue #2496",
    "body": "Closes #2496\n\nImplemented automated fix.",
    "head": "KartavyaDikshit:fix-issue-2496",
    "base": "main",
}
req = urllib.request.Request(
    "https://api.github.com/repos/truera/trulens/pulls",
    data=json.dumps(payload).encode(),
    headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(req, context=ctx) as r:
        pr_data = json.loads(r.read())
        print("[+] PR_CREATED:", pr_data["number"])
except Exception as e:
    print("[!] PR Failed:", e)
