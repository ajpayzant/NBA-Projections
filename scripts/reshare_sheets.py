"""
NBA Google Sheets Reshare — Apps Script trigger approach
=========================================================
The service account cannot reshare files it has lost access to.
The ONLY reliable solution is an Apps Script bound to the Season Index
sheet that runs as the file owner (your work account).

This script calls the Apps Script API to trigger that reshare function.

SETUP (one time):
1. Open the Season Index sheet
2. Extensions > Apps Script
3. Paste the script below and save
4. Deploy > New deployment > Web app > Execute as: Me > Access: Anyone
5. Copy the deployment ID and paste into APPS_SCRIPT_DEPLOYMENT_ID below

Apps Script code to paste:
---
function reshareWithServiceAccount() {
  var files = [
    DriveApp.getFileById('1fpBr-WiGRLFyWNdyq4BWRiiVtnFKqDcEmh8GdeRoiFQ'),
    DriveApp.getFolderById('17jTJJlZU779UViBC4mczamt7U29c5nSo'),
    DriveApp.getFolderById('1G8W9-KPTLQ6ujCWvgwsqY6M8J3PCRyYB'),
  ];
  var email = 'pll-projections-writer@pll-projections.iam.gserviceaccount.com';
  files.forEach(function(f) {
    try { f.addEditor(email); } catch(e) {}
  });
  return ContentService.createTextOutput('OK');
}

function doGet(e) {
  return reshareWithServiceAccount();
}

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('NBA Tools')
    .addItem('Reshare with Service Account', 'reshareWithServiceAccount')
    .addToUi();
}
---

Then set up a time-based trigger for reshareWithServiceAccount to run hourly.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SERVICE_ACCOUNT_EMAIL = "pll-projections-writer@pll-projections.iam.gserviceaccount.com"
SEASON_INDEX_ID       = "1fpBr-WiGRLFyWNdyq4BWRiiVtnFKqDcEmh8GdeRoiFQ"
SEASON_FOLDER_ID      = "17jTJJlZU779UViBC4mczamt7U29c5nSo"
NBA_ROOT_FOLDER_ID    = "1G8W9-KPTLQ6ujCWvgwsqY6M8J3PCRyYB"

# Set this after deploying the Apps Script web app
# Go to: Apps Script > Deploy > Manage deployments > copy Deployment ID
APPS_SCRIPT_DEPLOYMENT_ID = ""   # e.g. "AKfycb..."


def trigger_apps_script_reshare() -> bool:
    """
    Call the Apps Script web app deployment to reshare all NBA files.
    Returns True if the call succeeded.
    """
    if not APPS_SCRIPT_DEPLOYMENT_ID:
        return False
    try:
        import requests
        url = f"https://script.google.com/macros/s/{APPS_SCRIPT_DEPLOYMENT_ID}/exec"
        resp = requests.get(url, timeout=15)
        return resp.status_code == 200 and "OK" in resp.text
    except Exception:
        return False


def verify_access(verbose: bool = True) -> bool:
    """Verify the service account can read the Season Index sheet."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        from pathlib import Path
        import json

        # Load key from secrets.toml if available, else use hardcoded
        key_file = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
        if key_file.exists():
            import re
            text = key_file.read_text(encoding="utf-8")
            # Simple extraction of private_key from toml
            m = re.search(r'private_key\s*=\s*"(.*?)"', text, re.DOTALL)
            pk = m.group(1).replace("\\n", "\n") if m else None
            me = re.search(r'client_email\s*=\s*"(.*?)"', text)
            ce = me.group(1) if me else SERVICE_ACCOUNT_EMAIL
        else:
            pk = None; ce = SERVICE_ACCOUNT_EMAIL

        sa = {
            'type': 'service_account',
            'project_id': 'pll-projections',
            'private_key': pk or '',
            'client_email': ce,
            'token_uri': 'https://oauth2.googleapis.com/token',
        }
        creds = Credentials.from_service_account_info(
            sa, scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SEASON_INDEX_ID)
        if verbose:
            print(f"  [OK]  Sheet accessible: {sh.title}")
        return True
    except Exception as e:
        if verbose:
            print(f"  [FAIL] Sheet not accessible: {e}")
        return False


if __name__ == "__main__":
    print("NBA Google Sheets Access Check")
    print("=" * 40)
    ok = verify_access()
    if ok:
        print("\nAccess OK — no action needed.")
    else:
        print("\nAccess lost.")
        if APPS_SCRIPT_DEPLOYMENT_ID:
            print("Triggering Apps Script reshare...")
            success = trigger_apps_script_reshare()
            if success:
                print("Apps Script reshare triggered.")
                ok = verify_access()
                if ok:
                    print("Access restored.")
                else:
                    print("Still failing — try clicking NBA Tools > Reshare in the sheet.")
            else:
                print("Apps Script call failed.")
        print()
        print("MANUAL FIX:")
        print("1. Open the Season Index sheet")
        print("2. Click: NBA Tools > Reshare with Service Account")
        print(f"   (or share manually with {SERVICE_ACCOUNT_EMAIL})")
