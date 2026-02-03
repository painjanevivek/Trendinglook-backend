from fastapi import FastAPI
import requests
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------------
# APP + CORS
# --------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------
# SEC REQUIRED HEADERS
# --------------------------------------------------------
SEC_HEADERS = {
    "User-Agent": "TrendingLookApp/1.0 (contact: example@gmail.com)"
}

# --------------------------------------------------------
# GLOBAL TICKER STORAGE
# --------------------------------------------------------
TICKER_MAP = {}      # {"AAPL": "0000320193"}
TICKER_DATA = []     # [{"ticker": "AAPL", "name": "Apple Inc."}]


# --------------------------------------------------------
# LOAD TICKERS FROM SEC
# --------------------------------------------------------
def load_ticker_map():
    global TICKER_MAP, TICKER_DATA

    print("🔄 Loading SEC tickers...")

    url = "https://www.sec.gov/files/company_tickers.json"

    try:
        resp = requests.get(url, headers=SEC_HEADERS)
        print("Status:", resp.status_code)

        if resp.status_code != 200:
            print("❌ SEC API failed")
            return

        data = resp.json()

        for _, entry in data.items():
            ticker = entry["ticker"].upper()
            name = entry["title"]
            cik = str(entry["cik_str"]).zfill(10)

            TICKER_MAP[ticker] = cik
            TICKER_DATA.append({"ticker": ticker, "name": name})

        print("✅ Loaded", len(TICKER_DATA), "tickers")

    except Exception as e:
        print("❌ Error:", e)


load_ticker_map()


# --------------------------------------------------------
# HELPER — GET LATEST VALUE
# --------------------------------------------------------
def extract_latest_value(facts, tag):
    if tag not in facts:
        return None

    units_dict = facts[tag]["units"]
    first_key = list(units_dict.keys())[0]
    entries = units_dict[first_key]

    # sort by end date descending
    entries = sorted(entries, key=lambda x: x.get("end", ""), reverse=True)

    for item in entries:
        if "val" in item:
            return item["val"]

    return None


# --------------------------------------------------------
# HELPER — EXTRACT LAST 5 YEARS OF ANNUAL VALUES
# --------------------------------------------------------
def extract_last_5_years(facts, tag):
    """
    Returns list:
    [
        {"year": 2023, "value": 1234},
        {"year": 2022, "value": 1120},
        ...
    ]
    """
    if tag not in facts:
        return []

    units_dict = facts[tag].get("units", {})
    if not units_dict:
        return []

    first_key = list(units_dict.keys())[0]
    entries = units_dict[first_key]

    result = []

    for item in entries:
        form = item.get("form", "")
        end = item.get("end", "")
        val = item.get("val")

        # Only annual filings
        if "10-K" in form and val is not None and len(end) >= 4:
            year = int(end[:4])
            result.append({"year": year, "value": val})

    # sort: newest → oldest
    result.sort(key=lambda x: x["year"], reverse=True)

    return result[:5]


# --------------------------------------------------------
# HEALTH SCORE ENGINE
# --------------------------------------------------------
def compute_health_score(revenue, net_income, ocf, icf, fcf):
    score = 0
    details = {}

    # Profit margin
    if revenue and net_income:
        margin = net_income / revenue
        details["profit_margin"] = margin

        if margin > 0.20:
            score += 25
        elif margin > 0.10:
            score += 20
        elif margin > 0.05:
            score += 15
        elif margin > 0:
            score += 10
        else:
            score += 2

    # Cash conversion
    if ocf and net_income:
        conv = ocf / net_income if net_income != 0 else 0
        details["cash_conversion"] = conv

        if conv > 1.3:
            score += 25
        elif conv > 1.0:
            score += 20
        elif conv > 0.8:
            score += 15
        else:
            score += 5

    # Investment / CapEx behavior
    details["capex"] = icf
    if icf < 0:
        score += 10
    else:
        score += 5

    # Financing activity
    details["financing"] = fcf
    if fcf < 0:
        score += 15
    else:
        score += 5

    return min(100, score), details


# --------------------------------------------------------
# API — LIST OF TICKERS
# --------------------------------------------------------
@app.get("/tickers")
def get_tickers():
    return TICKER_DATA


# --------------------------------------------------------
# API — MAIN FINANCIALS + TRENDS (FULL)
# --------------------------------------------------------
@app.get("/financials/{ticker}")
def get_financials(ticker: str):
    ticker = ticker.upper()

    if ticker not in TICKER_MAP:
        return {"error": "Ticker not found"}

    cik = TICKER_MAP[ticker]

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    data = requests.get(url, headers=SEC_HEADERS).json()

    facts = data["facts"]["us-gaap"]

    # Latest values
    revenue = extract_latest_value(facts, "Revenues")
    net_income = extract_latest_value(facts, "NetIncomeLoss")
    ocf = extract_latest_value(facts, "NetCashProvidedByUsedInOperatingActivities")
    icf = extract_latest_value(facts, "NetCashProvidedByUsedInInvestingActivities")
    fcf = extract_latest_value(facts, "NetCashProvidedByUsedInFinancingActivities")

    # Trends (5-year history)
    rev_hist = extract_last_5_years(facts, "Revenues")
    ni_hist = extract_last_5_years(facts, "NetIncomeLoss")
    ocf_hist = extract_last_5_years(facts, "NetCashProvidedByUsedInOperatingActivities")
    icf_hist = extract_last_5_years(facts, "NetCashProvidedByUsedInInvestingActivities")

    # Compute FCF trend = OCF - abs(ICF)
    fcf_hist = []
    for i in range(min(len(ocf_hist), len(icf_hist))):
        fcf_hist.append({
            "year": ocf_hist[i]["year"],
            "value": ocf_hist[i]["value"] - abs(icf_hist[i]["value"])
        })

    # Health Score
    health_score, indicators = compute_health_score(
        revenue, net_income, ocf, icf, fcf
    )

    return {
        "ticker": ticker,

        # Latest values
        "revenue": revenue,
        "net_income": net_income,
        "operating_cash_flow": ocf,
        "investing_cash_flow": icf,
        "financing_cash_flow": fcf,
        "net_cash_flow": (ocf or 0) + (icf or 0) + (fcf or 0),

        # Trend block used by frontend
        "trends": {
            "revenue": rev_hist,
            "net_income": ni_hist,
            "operating_cash_flow": ocf_hist,
            "free_cash_flow": fcf_hist
        },

        # Health
        "health_score": health_score,
        "indicators": indicators
    }


@app.get("/api/financials/sankey/{ticker}")
def get_financials_sankey(ticker: str):
    """Fetch latest 10-K via edgartools, extract an income statement, and return Sankey-format nodes/links.

    The function is synchronous (def) because edgartools is blocking and should run in FastAPI's threadpool.
    Returns 404 if the ticker is not known or if the income statement cannot be located.
    """
    ticker = ticker.upper()

    # Ticker validation
    if ticker not in TICKER_MAP:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Ticker not found")

    try:
        import edgartools
        import pandas as pd
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Dependency error: {e}")

    try:
        income_df = None

        # Try several plausible edgartools APIs (be tolerant to different library versions)
        if hasattr(edgartools, "get_income_statement"):
            income_df = edgartools.get_income_statement(ticker, form="10-K", latest=True)

        elif hasattr(edgartools, "Company"):
            # Many versions expose a Company helper
            try:
                company = edgartools.Company(ticker)
                if hasattr(company, "get_income_statement"):
                    income_df = company.get_income_statement(form="10-K", latest=True)
                elif hasattr(company, "get_financials"):
                    fin = company.get_financials()
                    if isinstance(fin, dict):
                        income_df = fin.get("income_statement")
            except Exception:
                income_df = None

        elif hasattr(edgartools, "get_filing"):
            filing = edgartools.get_filing(ticker, form="10-K", latest=True)
            # try to find an income table in the filing
            if filing:
                tables = None
                if isinstance(filing, dict):
                    tables = filing.get("tables")
                else:
                    tables = getattr(filing, "tables", None)

                if tables:
                    for tbl in tables:
                        name = ""
                        data = None
                        if isinstance(tbl, dict):
                            name = (tbl.get("name") or "").lower()
                            data = tbl.get("data")
                        else:
                            name = (getattr(tbl, "name", "") or "").lower()
                            data = getattr(tbl, "data", None)

                        if "income" in name or "operations" in name or "statement of operations" in name:
                            income_df = pd.DataFrame(data)
                            break

        if income_df is None:
            raise ValueError("Income statement not found in latest 10-K")

        # Normalize to DataFrame
        income_df = pd.DataFrame(income_df)

        # If first column contains row labels, use it as index
        first_col_name = income_df.columns[0].lower() if income_df.columns.size > 0 else ""
        if first_col_name in ("line", "label", "description", "item", "account"):
            income_df = income_df.set_index(income_df.columns[0])

        # Ensure index are strings
        income_df.index = income_df.index.astype(str)

        # Find the latest numeric column (prefer the right-most numeric column)
        numeric_cols = []
        for c in income_df.columns:
            try:
                s = pd.to_numeric(income_df[c], errors="coerce")
                if s.notna().any():
                    numeric_cols.append(c)
            except Exception:
                continue

        if numeric_cols:
            latest_col = numeric_cols[-1]
        elif "value" in [c.lower() for c in income_df.columns]:
            latest_col = [c for c in income_df.columns if c.lower() == "value"][0]
        else:
            raise ValueError("No numeric columns found in income statement")

        def find_amount(patterns):
            """Return the numeric value for the first row whose label contains one of the patterns (case-insensitive)."""
            patl = [p.lower() for p in patterns]
            for label in income_df.index.astype(str):
                ll = label.lower()
                for p in patl:
                    if p in ll:
                        try:
                            val = pd.to_numeric(income_df.loc[label, latest_col], errors="coerce")
                            if not pd.isna(val):
                                return float(val)
                        except Exception:
                            continue
            return None

        # Candidate field patterns
        revenue = find_amount(["revenue", "total revenue", "revenues"])
        cost = find_amount(["cost of revenue", "cost of goods sold", "cogs"])
        gross = find_amount(["gross profit", "gross margin"]) or (revenue - cost if revenue is not None and cost is not None else None)
        r_and_d = find_amount(["research and development", "r&d", "research and development expense"])
        sga = find_amount(["selling, general", "selling general", "sg&a", "selling general and administrative"])
        operating_income = find_amount(["operating income", "income from operations", "operating profit"])
        tax = find_amount(["income tax expense", "provision for income taxes", "tax provision"])
        net_income = find_amount(["net income", "net loss"])

        # fallback: compute net income if possible
        if net_income is None and operating_income is not None and tax is not None:
            net_income = operating_income - tax

        # Treat missing values as 0 for visualization; keep the raw values too
        vals = {
            "Revenue": revenue or 0.0,
            "Cost of Revenue": cost or 0.0,
            "Gross Profit": gross or 0.0,
            "R&D": r_and_d or 0.0,
            "SG&A": sga or 0.0,
            "Operating Income": operating_income or 0.0,
            "Tax Provision": tax or 0.0,
            "Net Income": net_income or 0.0,
        }

        # Build nodes and links
        nodes = [{"name": name} for name in vals.keys()]
        idx_map = {name: i for i, name in enumerate(vals.keys())}
        links = []

        def add_link(src, tgt, amount):
            if amount is None or amount == 0:
                return
            links.append({"source": idx_map[src], "target": idx_map[tgt], "value": float(amount)})

        # Mapping flows
        add_link("Revenue", "Cost of Revenue", vals["Cost of Revenue"])
        add_link("Revenue", "Gross Profit", vals["Gross Profit"])
        add_link("Gross Profit", "R&D", vals["R&D"])
        add_link("Gross Profit", "SG&A", vals["SG&A"])
        add_link("Gross Profit", "Operating Income", vals["Operating Income"])
        add_link("Operating Income", "Tax Provision", vals["Tax Provision"])
        add_link("Operating Income", "Net Income", vals["Net Income"])

        return {"nodes": nodes, "links": links, "raw_values": vals}

    except Exception as e:
        from fastapi import HTTPException
        # For missing filings or parsing problems return 404; for other errors this becomes a 404 with details
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/")
def home():
    return {"status": "TrendingLook Backend Running"}


if __name__ == "__main__":
    # Start Uvicorn when running `python main.py`
    import os
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("RELOAD", "true").lower() in ("1", "true", "yes")

    print(f"Starting TrendingLook backend on {host}:{port} (reload={reload})")
    uvicorn.run("main:app", host=host, port=port, reload=reload)
