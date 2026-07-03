"""NBA Projection System — Landing Page"""
import streamlit as st

st.set_page_config(
    page_title="NBA Projections",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
  .nav-card {
    background: rgba(30,41,59,.6);
    border: 1px solid rgba(148,163,184,.15);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color .2s;
  }
  .nav-card:hover { border-color: #c9a227; }
  .nav-title { font-size: 1.1rem; font-weight: 700; color: #f1f5f9; margin-bottom: 4px; }
  .nav-desc  { font-size: .85rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

st.title("🏀 NBA Projection System")
st.markdown("**Player and game outcome projections powered by EWM rate modeling + Monte Carlo simulation.**")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="nav-card">
      <div class="nav-title">1 · Projections</div>
      <div class="nav-desc">Select a game, configure settings, and run projections for both teams and all active players.</div>
    </div>
    <div class="nav-card">
      <div class="nav-title">2 · Depth Charts</div>
      <div class="nav-desc">Manage lineups — set active/inactive, injury ratings, minutes overrides, and rating adjustments per player.</div>
    </div>
    <div class="nav-card">
      <div class="nav-title">3 · Player Props</div>
      <div class="nav-desc">Browse prop markets for PTS, REB, AST, 3PM, BLK, STL and combinations. Compare fair odds to market.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="nav-card">
      <div class="nav-title">4 · Game Lines</div>
      <div class="nav-desc">Moneyline, spread, and totals with fair probability estimates and market comparison.</div>
    </div>
    <div class="nav-card">
      <div class="nav-title">5 · Projection History</div>
      <div class="nav-desc">Track saved projections vs actuals. Review model accuracy by stat, position, and player.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<span style="font-size:.75rem;color:#64748b;">'
    'Data via nba_api · EWM rate model with RF team correction · 20,000-sim Monte Carlo · '
    'Projections are for informational purposes only.'
    '</span>',
    unsafe_allow_html=True,
)
