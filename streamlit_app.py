import streamlit as st
import requests

API_URL = "https://recommendation-api-production-9231.up.railway.app"

st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Product Recommendation System")
st.markdown("Real-time recommendations using ALS Collaborative Filtering")

# ── Sidebar ───────────────────────────────────────────
st.sidebar.header("About")
st.sidebar.info("""
**Stack:**
- FastAPI + ALS model
- 7.8M Amazon interactions
- Cold start handling
- Redis caching
- Docker + Railway
""")

st.sidebar.header("Try these users")
st.sidebar.code("A1N63KPEPN5HVU")
st.sidebar.code("A2SUAM1J3GNN3B")
st.sidebar.code("unknownuser123")

# ── Main ──────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    user_id = st.text_input(
        "Enter User ID",
        value="A1N63KPEPN5HVU",
        placeholder="Enter any user ID..."
    )
with col2:
    n = st.number_input("# Recs", min_value=1,
                        max_value=20, value=5)

if st.button("Get Recommendations 🚀", type="primary"):
    with st.spinner("Fetching recommendations..."):
        try:
            # Get recommendations
            resp = requests.get(
                f"{API_URL}/recommend/{user_id}",
                params={"n": n},
                timeout=30
            )

            if resp.status_code == 200:
                data = resp.json()

                # Show metadata
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Recs", data['total'])
                col2.metric("Model", data['model_version'])
                col3.metric("Served From", data['served_from'])

                # Color code served_from
                if data['served_from'] == 'als_model':
                    st.success("✓ Personalized recommendations from ALS model")
                elif data['served_from'] == 'cache':
                    st.info("⚡ Served from Redis cache")
                else:
                    st.warning("⚠ Cold start — showing popular items")

                # Show recommendations
                st.subheader("Recommended Products")
                for i, item in enumerate(data['recommendations']):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{i+1}.** `{item['item_id']}`")
                    with col2:
                        st.write(f"Score: {item['score']}")
                    st.divider()

            else:
                st.error(f"API Error: {resp.status_code}")

        except Exception as e:
            st.error(f"Connection error: {e}")

# ── Health Check ──────────────────────────────────────
st.subheader("API Health")
if st.button("Check API Status"):
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        data = resp.json()

        col1, col2, col3 = st.columns(3)
        col1.metric("Status", data['status'])
        col2.metric("Model Loaded",
                    "✅" if data['model_loaded'] else "❌")
        col3.metric("Redis",
                    "✅" if data['redis_connected'] else "❌")
    except Exception as e:
        st.error(f"API not reachable: {e}")

# ── Footer ────────────────────────────────────────────
st.markdown("---")
st.markdown("""
**Project:** Real-Time Product Recommendation API  
**GitHub:** [recommendation-api](https://github.com/atharva-pawar80/recommendation-api)  
**Live API:** [Swagger Docs](https://recommendation-api-production-9231.up.railway.app/docs)
""")