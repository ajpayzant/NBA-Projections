"""Quick end-to-end test: project BOS vs LAL and save to Google Sheets."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Patch st.secrets
import types, streamlit as st
mock = types.SimpleNamespace()
mock.__getitem__ = lambda self, k: {
    'gcp_service_account': {
        'type':'service_account','project_id':'pll-projections',
        'private_key_id':'ebad31a46340288bd847b6cff60c3816b92a70de',
        'private_key':'-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCz4/Yxq8h0SEPT\nmvm3CTPXu5QguuoupVoR8YVAVozP+fQ+DFsHnuP7C130uecjO7S8FT3nigpiufgJ\n64LN6ujtq9mSI5bvWqMGAvHuPfVusR7TxXbTGzGCIYK5IWLr/YB3DOBHJ4WBgcy+\nWzaP4ojtecC6cKrhFrggpxIFE3Q9nvGUtnUhMBuWJ0EyfIw3m734cCHCuXMSePMT\n3Sn3og7SfjwVfVVI7Dfgra0pZqCa2TJ6WL7o7atxYzrm9ufZDqmnJlDB/VJy0xfv\nRw5BHBSl1G3d0VQRVx3SL/eGk6ZhqcyXQYGOdidcGqbsWrGMuq/7rhqlZCJc/xez\ne2BNmzUBAgMBAAECggEAPMXUTBaRS3uIcooGN+hje8vysJYE/Io/IhW6ozA6NR7n\n/ThWXn423f6KmN1fMV9/4RS6C6ludckG+271i1SDqZunjr/4Q1eXALZo/kgvTNyI\nohmxWYNz19czXxDg5wIG7vFKKa+34jc0MNEG31g0UyoRSss6Z32x6dIXe+CFIjWp\nTj+AM2HfVFO/Hi4qhpmIgh7DRcG0CpALk2ESiU2mLBVZ+29u51B0ddzjgn28oTbq\nGuLkVCj1l2G09dPMNyp0l1BzokiLYnijUGiUnj1UC73kO0ax/u5yOyMWkCEIEqyX\n9xe9i0uWRvmDWijj44maXZ41/mr/wykHgmAcMw5NmwKBgQDoA9HFnTF6tLCwh+59\nPHUNZCBCXrCHsp6tppl04aLOr5tQkWraSoTppLPXy1Zg1X7Oe4OImFHtKlUiF9Of\ndYO+GkXc9WUeYl6F4wNb/m8JCfKiipPyveVm72Xd2d2Gyzc4qyXeJwmEwLyh2oQO\nDEvX1ppNgEEEGGJf3fvzZqLw0wKBgQDGfLCmKbI/8C/oPF9TglD6eruJzImFJyMw\nSpksN0z/SQH82ztAcynb89JdxB98sz7kKB9mjZjrStOn7fvQSJgfHbPfptRMMfCt\n5serrvciepC5U363gvMRdzm0iOa07r+lbQEri+cQaOBL9pmg46+OGgdKFpMDL87O\nE1me8WG+WwKBgG5AumOE0ml0Ce6pebxLiXgml2nSo2Koj90HKS8wtfQv4MeFgthe\nxxBcMxYdy1tSuOKYMSYs8+mWz0PXPLou1r70rzRT7IxmxHItMYB6xPrvnjx9S9bz\nRFsI8khdanbOhNxKAEG0HULXcAwd0dj3IOddVI/1nW+7wqu5yiudH1r/AoGAHPsd\ny9Uwupc2V4FhJc9URY5gDZm4xqFVSPrLbKJScr/VM3dLKnjmNsBeCeTV+B4v455c\nH1wzZL+TMeTUrK+8zmZG2jQAsXNlQe79Xnr4iKc+tGCVkvPiy70NxudqUCbAAsZs\nslAGF+ZIQa8q9UvpWSVBxTaQlpmHZ515q3RxIhsCgYAS8BcXv1tKcKABKo4Azccx\nZd2ZEMYY7pmScHCRqEfXrphrTbctMSGnDLuLqBl1vqe0MNGxSQIWcAgOeNppYqWt\n7vUBb6Bn8tP33Ti9uYVhNWrbz1VQPBsudSgaySzNy/zWPRFnCTvnlyqzrv654hJG\nrs+1TpltwUojxAZXuX5/CQ==\n-----END PRIVATE KEY-----\n',
        'client_email':'pll-projections-writer@pll-projections.iam.gserviceaccount.com',
        'client_id':'117082471257917827889',
        'auth_uri':'https://accounts.google.com/o/oauth2/auth',
        'token_uri':'https://oauth2.googleapis.com/token',
    },
    'google_drive': {
        'projections_folder_id': '1G8W9-KPTLQ6ujCWvgwsqY6M8J3PCRyYB',
        'nba_sheet_id': '1fpBr-WiGRLFyWNdyq4BWRiiVtnFKqDcEmh8GdeRoiFQ',
    },
}[k]
st.secrets = mock

from nba_engine import NBAProjectionEngine
import gsheets_writer_nba as gw

print("Loading engine...")
eng = NBAProjectionEngine()
eng.load()
eng.fit()

game = {'away_team_abbr': 'LAL', 'home_team_abbr': 'BOS',
        'game_date': '2025-10-22', 'game_id': 'LAL@BOS_2025-10-22'}

print("Running projection...")
result = eng.project('BOS', 'LAL', game_date='2025-10-22')

print("Saving to Google Sheets...")
tab = gw.save_snapshot(result, game, eng)
print(f'Saved tab: {tab}')
print(f'Review: https://docs.google.com/spreadsheets/d/1fpBr-WiGRLFyWNdyq4BWRiiVtnFKqDcEmh8GdeRoiFQ')
