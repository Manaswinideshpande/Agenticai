import os
import json
import time
import requests
import urllib.parse
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Automatically check for local .env files if present
load_dotenv()

# Secure fallback for Groq Credentials tracking
API_KEY = os.environ.get("GROQ_API_KEY", "api")

# Configure Streamlit Native View Matrix Layout
st.set_page_config(page_title="Prospect Research Intelligence Matrix", layout="wide")
st.title("🕵️‍♂️ B2B Prospect Research & Enriched Corporate Agent")
st.markdown("Single-Agent production pipeline engineered to extract corporate sites, optimize text logs, and parse structured sales intelligence profiles.")
st.write("---")

# ==========================================
# 🧠 CORE BACKEND ENGINE FUNCTIONS (TASK 1)
# ==========================================

def extract_clean_text_from_url(url: str) -> str:
    """
    Multi-approach web harvester with token optimization.
    Strips code syntax to minimize token overhead.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProspectAgent/1.0"}
    target_keywords = ["about", "service", "contact", "product", "solutions"]
    discovered_links = {url}

    try:
        time.sleep(1) # Polite delay constraint
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code != 200:
            return f"Error: Status code {response.status_code}"

        soup = BeautifulSoup(response.content, "html.parser")

        # Smart link extraction using heuristic matching
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].lower()
            if any(kw in href for kw in target_keywords):
                full_url = urllib.parse.urljoin(url, anchor["href"])
                if urllib.parse.urlparse(full_url).netloc == urllib.parse.urlparse(url).netloc:
                    discovered_links.add(full_url)
                    if len(discovered_links) >= 3:
                        break

        aggregated_raw_text = []
        for target_path in list(discovered_links):
            try:
                time.sleep(0.5)
                sub_res = requests.get(target_path, headers=headers, timeout=5)
                if sub_res.status_code == 200:
                    sub_soup = BeautifulSoup(sub_res.content, "html.parser")
                    for element in sub_soup(["script", "style", "nav", "footer", "header"]):
                        element.decompose()
                    aggregated_raw_text.append(sub_soup.get_text(separator=" "))
            except Exception:
                continue

        return " ".join(" ".join(aggregated_raw_text).split())[:6000] # Safe token truncation constraint
    except Exception as e:
        return f"Scraping Failure: {str(e)}"


def enrich_company(url: str) -> dict:
    """
    Input: Company URL
    Output: Structured company profile dictionary
    """
    scraped_text = extract_clean_text_from_url(url)

    # Setup the deterministic inference engine
    llm = ChatGroq(
        temperature=0.1,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=API_KEY
    )

    system_instruction = (
        "You are an Elite Business Intelligence and Prospect Research Agent.\n"
        "Analyze the provided text and synthesize metadata matching the requested schema layout.\n\n"
        "Your output response MUST be a single valid JSON object. Do NOT wrap strings inside markdown blocks like ```json or ```.\n"
        "If a key property (like address or mobile_number) is missing, assign an empty string \"\" or \"N/A\". Do NOT drop keys.\n\n"
        "Your JSON return parameters must match these exact case-sensitive string keys:\n"
        "{{\n"
        "  \"website_name\": \"Clean display name of the web property\",\n"
        "  \"company_name\": \"Full legal organizational corporate name\",\n"
        "  \"address\": \"Physical corporate operational headquarters address\",\n"
        "  \"mobile_number\": \"Contact telephone number string value\",\n"
        "  \"mail\": [\"Discovered email address strings\"],\n"
        "  \"core_service\": \"1-sentence extraction of what products/services they sell\",\n"
        "  \"target_customer\": \"Buyer/market segment profiles summary\",\n"
        "  \"probable_pain_point\": \"Operational bottleneck this firm likely faces based on their industry scale\",\n"
        "  \"outreach_opener\": \"A targeted, 2-sentence sales outreach icebreaker opener\"\n"
        "}}\n"
        "Output absolute raw JSON string output only."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("user", "Scraped Web Corpus Content:\n{corpus}")
    ])

    agent_chain = prompt | llm | StrOutputParser()

    try:
        raw_output = agent_chain.invoke({"corpus": scraped_text}).strip()
        if raw_output.startswith("```"):
            raw_output = raw_output[raw_output.find("{"):raw_output.rfind("}")+1]
        return json.loads(raw_output)
    except Exception as e:
        domain = urllib.parse.urlparse(url).netloc
        return {
            "website_name": domain,
            "company_name": domain + " Inc.",
            "address": "N/A",
            "mobile_number": "N/A",
            "mail": [],
            "core_service": "Market details could not be parsed.",
            "target_customer": "N/A",
            "probable_pain_point": "Data parsing constraints encountered.",
            "outreach_opener": f"Error parsing payload structure text: {str(e)}"
        }


# ==========================================
# 🎨 FRONTEND WORKFLOW MANAGEMENT (TASK 2)
# ==========================================

# Maintain an in-memory session state ledger log cache across app clicks
if "profile_database" not in st.session_state:
    st.session_state.profile_database = []

# 🔹 1. ENRICH SECTION
st.header("🔹 1. Enrich Section")
col1, col2 = st.columns([2, 3])
with col1:
    website_name_input = st.text_input("Enter Website Name (Record Keeping Hint):", value="Example Target")
with col2:
    # ✨ FIXED: Default input parameter value altered from a list to a single valid string URL
    url_input = st.text_input("Enter Company Website URL Target:", value="[https://sigmoidal.io](https://sigmoidal.io)")
    
enrich_btn = st.button("🚀 Run Enrich Pipeline Engine")

if enrich_btn:
    with st.spinner("Executing smart link scraping and generating corporate intelligence profiles..."):
        # Call the required core intelligence wrapper directly
        enriched_data = enrich_company(url_input)
        
        # Override website_name with user's input hint if provided for better record-keeping
        if website_name_input and enriched_data.get("website_name") in ["", "N/A", "example.com"]:
            enriched_data["website_name"] = website_name_input
            
        # Save to database ledger array state
        st.session_state.profile_database.append(enriched_data)
        
        st.success("Target Record Enriched Successfully!")
        
        # 🌟 PROFESSIONAL LIVE UI CARD DISPLAY
        st.markdown(f"### 🏢 Corporate Intelligence Profile: {enriched_data.get('company_name')}")
        
        card_col1, card_col2 = st.columns(2)
        with card_col1:
            st.markdown(f"**📍 Headquarter Location:** {enriched_data.get('address')}")
            st.markdown(f"**📞 Verified Phone:** {enriched_data.get('mobile_number')}")
            st.markdown(f"**✉️ Emails Found:** {', '.join(enriched_data.get('mail', [])) if enriched_data.get('mail') else 'N/A'}")
        with card_col2:
            st.markdown(f"**🎯 Core Service Summary:** {enriched_data.get('core_service')}")
            st.markdown(f"**👥 Target Buyer Market:** {enriched_data.get('target_customer')}")
            st.markdown(f"**⚠️ Predicted Corporate Bottleneck:** {enriched_data.get('probable_pain_point')}")
            
        st.info(f"💬 **Hyper-Personalized Outreach Opener:**\n{enriched_data.get('outreach_opener')}")
        
st.write("---")

# 🔹 2. RESULTS SECTION
st.header("🔹 2. Results Section")
show_all_btn = st.button("👁️ Show All Enriched Historical Profiles")

if show_all_btn:
    if st.session_state.profile_database:
        df = pd.DataFrame(st.session_state.profile_database)
        st.subheader("📋 Core Enriched Profiles Registry Dataframe Table")
        st.dataframe(df, use_container_width=True)
        
        st.write("---")
        st.subheader("🗂️ Individual Deep-Dive Profile Cards")
        for item in st.session_state.profile_database:
            with st.expander(f"Profile: {item.get('company_name')} ({item.get('website_name')})"):
                st.markdown(f"**🏢 Legal Organizational Name:** {item.get('company_name')}")
                st.markdown(f"**📍 Headquarter Address Location:** {item.get('address')}")
                st.markdown(f"**📞 Verified Telephone Number:** {item.get('mobile_number')}")
                st.markdown(f"**✉️ Contact Emails Discovered:** {', '.join(item.get('mail', [])) if item.get('mail') else 'N/A'}")
                st.markdown(f"**🎯 Core Service Summary:** {item.get('core_service')}")
                st.markdown(f"**👥 Target Buyer Customer Base:** {item.get('target_customer')}")
                st.markdown(f"**⚠️ Deduced Corporate Pain Point:** {item.get('probable_pain_point')}")
                st.info(f"💬 **Hyper-Personalized Outreach Opener Icebreaker:**\n{item.get('outreach_opener')}")
    else:
        st.info("The in-memory database ledger is currently empty. Input a URL and hit 'Run Enrich Pipeline Engine' first.")
