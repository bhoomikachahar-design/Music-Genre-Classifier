"""
Runner script for Music Genre Classifier & Explorer
Launches FastAPI backend on http://127.0.0.1:8000 and opens the browser.
"""

import sys
import os
import time
import webbrowser
import uvicorn

def main():
    print("=" * 65)
    print("      MUSIC GENRE CLASSIFIER & EXPLORER")
    print("   GLA University - Department of Computer Engineering & Applications")
    print("=" * 65)
    print("\n[+] Starting FastAPI Application on http://127.0.0.1:8000 ...")
    print("[+] Press Ctrl+C in this terminal to stop the server.\n")

    # Open browser automatically after a short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Run Uvicorn server
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
