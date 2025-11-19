import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

# ML imports (optional)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Nifty500 Multi-Timeframe Buy/Sell/Hold", layout="wide")
st.title("📊 Nifty500 Multi-Timeframe Buy/Sell/Hold — 1H / Daily / Weekly")

# ---------------- TICKERS ----------------
NIFTY500_TICKERS = [
    "360ONE.NS","3MINDIA.NS","ABB.NS","TIPSMUSIC.NS","ACC.NS","ACMESOLAR.NS","AIAENG.NS","APLAPOLLO.NS","AUBANK.NS","AWL.NS","AADHARHFC.NS",
    "AARTIIND.NS","AAVAS.NS","ABBOTINDIA.NS","ACE.NS","ADANIENSOL.NS","ADANIENT.NS","ADANIGREEN.NS","ADANIPORTS.NS","ADANIPOWER.NS","ATGL.NS",
    "ABCAPITAL.NS","ABFRL.NS","ABREL.NS","ABSLAMC.NS","AEGISLOG.NS","AFCONS.NS","AFFLE.NS","AJANTPHARM.NS","AKUMS.NS","APLLTD.NS",
    "ALIVUS.NS","ALKEM.NS","ALKYLAMINE.NS","ALOKINDS.NS","ARE&M.NS","AMBER.NS","AMBUJACEM.NS","ANANDRATHI.NS","ANANTRAJ.NS","ANGELONE.NS",
    "APARINDS.NS","APOLLOHOSP.NS","APOLLOTYRE.NS","APTUS.NS","ASAHIINDIA.NS","ASHOKLEY.NS","ASIANPAINT.NS","ASTERDM.NS","ASTRAZEN.NS","ASTRAL.NS",
    "ATUL.NS","AUROPHARMA.NS","AIIL.NS","DMART.NS","AXISBANK.NS","BASF.NS","BEML.NS","BLS.NS","BSE.NS","BAJAJ-AUTO.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","BAJAJHLDNG.NS","BAJAJHFL.NS","BALKRISIND.NS","BALRAMCHIN.NS","BANDHANBNK.NS","BANKBARODA.NS","BANKINDIA.NS","MAHABANK.NS",
    "BATAINDIA.NS","BAYERCROP.NS","BERGEPAINT.NS","BDL.NS","BEL.NS","BHARATFORG.NS","BHEL.NS","BPCL.NS","BHARTIARTL.NS","BHARTIHEXA.NS",
    "BIKAJI.NS","BIOCON.NS","BSOFT.NS","BLUEDART.NS","BLUESTARCO.NS","BBTC.NS","BOSCHLTD.NS","FIRSTCRY.NS","BRIGADE.NS","BRITANNIA.NS",
    "MAPMYINDIA.NS","CCL.NS","CESC.NS","CGPOWER.NS","CRISIL.NS","CAMPUS.NS","CANFINHOME.NS","CANBK.NS","CAPLIPOINT.NS","CGCL.NS",
    "CARBORUNIV.NS","CASTROLIND.NS","CEATLTD.NS","CENTRALBK.NS","CDSL.NS","CENTURYPLY.NS","CERA.NS","CHALET.NS","CHAMBLFERT.NS","CHENNPETRO.NS",
    "CHOLAHLDNG.NS","CHOLAFIN.NS","CIPLA.NS","CUB.NS","CLEAN.NS","COALINDIA.NS","COCHINSHIP.NS","COFORGE.NS","COHANCE.NS","COLPAL.NS",
    "CAMS.NS","CONCORDBIO.NS","CONCOR.NS","COROMANDEL.NS","CRAFTSMAN.NS","CREDITACC.NS","CROMPTON.NS","CUMMINSIND.NS","CYIENT.NS","DCMSHRIRAM.NS",
    "DLF.NS","DOMS.NS","DABUR.NS","DALBHARAT.NS","DATAPATTNS.NS","DEEPAKFERT.NS","DEEPAKNTR.NS","DELHIVERY.NS","DEVYANI.NS","DIVISLAB.NS",
    "DIXON.NS","LALPATHLAB.NS","DRREDDY.NS","DUMMYDBRLT.NS","EIDPARRY.NS","EIHOTEL.NS","EICHERMOT.NS","ELECON.NS","ELGIEQUIP.NS","EMAMILTD.NS",
    "EMCURE.NS","ENDURANCE.NS","ENGINERSIN.NS","ERIS.NS","ESCORTS.NS","ETERNAL.NS","EXIDEIND.NS","NYKAA.NS","FEDERALBNK.NS","FACT.NS",
    "FINCABLES.NS","FINPIPE.NS","FSL.NS","FIVESTAR.NS","FORTIS.NS","GAIL.NS","GVT&D.NS","GMRAIRPORT.NS","GRSE.NS","GICRE.NS",
    "GILLETTE.NS","GLAND.NS","GLAXO.NS","GLENMARK.NS","MEDANTA.NS","GODIGIT.NS","GPIL.NS","GODFRYPHLP.NS","GODREJAGRO.NS","GODREJCP.NS",
    "GODREJIND.NS","GODREJPROP.NS","GRANULES.NS","GRAPHITE.NS","GRASIM.NS","GRAVITA.NS","GESHIP.NS","FLUOROCHEM.NS","GUJGASLTD.NS","GMDCLTD.NS",
    "GNFC.NS","GPPL.NS","GSPL.NS","HEG.NS","HBLENGINE.NS","HCLTECH.NS","HDFCAMC.NS","HDFCBANK.NS","HDFCLIFE.NS","HFCL.NS",
    "HAPPSTMNDS.NS","HAVELLS.NS","HEROMOTOCO.NS","HSCL.NS","HINDALCO.NS","HAL.NS","HINDCOPPER.NS","HINDPETRO.NS","HINDUNILVR.NS","HINDZINC.NS",
    "POWERINDIA.NS","HOMEFIRST.NS","HONASA.NS","HONAUT.NS","HUDCO.NS","HYUNDAI.NS","ICICIBANK.NS","ICICIGI.NS","ICICIPRULI.NS","IDBI.NS",
    "IDFCFIRSTB.NS","IFCI.NS","IIFL.NS","INOXINDIA.NS","IRB.NS","IRCON.NS","ITC.NS","ITI.NS","INDGN.NS","INDIACEM.NS",
    "INDIAMART.NS","INDIANB.NS","IEX.NS","INDHOTEL.NS","IOC.NS","IOB.NS","IRCTC.NS","IRFC.NS","IREDA.NS","IGL.NS",
    "INDUSTOWER.NS","INDUSINDBK.NS","NAUKRI.NS","INFY.NS","INOXWIND.NS","INTELLECT.NS","INDIGO.NS","IGIL.NS","IKS.NS","IPCALAB.NS",
    "JBCHEPHARM.NS","JKCEMENT.NS","JBMA.NS","JKTYRE.NS","JMFINANCIL.NS","JSWENERGY.NS","JSWHL.NS","JSWINFRA.NS","JSWSTEEL.NS","JPPOWER.NS",
    "J&KBANK.NS","JINDALSAW.NS","JSL.NS","JINDALSTEL.NS","JIOFIN.NS","JUBLFOOD.NS","JUBLINGREA.NS","JUBLPHARMA.NS","JWL.NS","JUSTDIAL.NS",
    "JYOTHYLAB.NS","JYOTICNC.NS","KPRMILL.NS","KEI.NS","KNRCON.NS","KPITTECH.NS","KAJARIACER.NS","KPIL.NS","KALYANKJIL.NS","KANSAINER.NS",
    "KARURVYSYA.NS","KAYNES.NS","KEC.NS","KFINTECH.NS","KIRLOSBROS.NS","KIRLOSENG.NS","KOTAKBANK.NS","KIMS.NS","LTF.NS","LTTS.NS",
    "LICHSGFIN.NS","LTFOODS.NS","LTIM.NS","LT.NS","LATENTVIEW.NS","LAURUSLABS.NS","LEMONTREE.NS","LICI.NS","LINDEINDIA.NS","LLOYDSME.NS",
    "LODHA.NS","LUPIN.NS","MMTC.NS","MRF.NS","MGL.NS","MAHSEAMLES.NS","M&MFIN.NS","M&M.NS","MANAPPURAM.NS","MRPL.NS",
    "MANKIND.NS","MARICO.NS","MARUTI.NS","MASTEK.NS","MFSL.NS","MAXHEALTH.NS","MAZDOCK.NS","METROPOLIS.NS","MINDACORP.NS","MSUMI.NS",
    "MOTILALOFS.NS","MPHASIS.NS","MCX.NS","MUTHOOTFIN.NS","NATCOPHARM.NS","NBCC.NS","NCC.NS","NHPC.NS","NLCINDIA.NS","NMDC.NS",
    "NSLNISP.NS","NTPCGREEN.NS","NTPC.NS","NH.NS","NATIONALUM.NS","NAVA.NS","NAVINFLUOR.NS","NESTLEIND.NS","NETWEB.NS","NETWORK18.NS",
    "NEULANDLAB.NS","NEWGEN.NS","NAM-INDIA.NS","NIVABUPA.NS","NUVAMA.NS","OBEROIRLTY.NS","ONGC.NS","OIL.NS","OLAELEC.NS","OLECTRA.NS",
    "PAYTM.NS","OFSS.NS","POLICYBZR.NS","PCBL.NS","PGEL.NS","PIIND.NS","PNBHOUSING.NS","PNCINFRA.NS","PTCIL.NS","PVRINOX.NS",
    "PAGEIND.NS","PATANJALI.NS","PERSISTENT.NS","PETRONET.NS","PFIZER.NS","PHOENIXLTD.NS","PIDILITIND.NS","PEL.NS","PPLPHARMA.NS","POLYMED.NS",
    "POLYCAB.NS","POONAWALLA.NS","PFC.NS","POWERGRID.NS","PRAJIND.NS","PREMIERENE.NS","PRESTIGE.NS","PNB.NS","RRKABEL.NS","RBLBANK.NS",
    "RECLTD.NS","RHIM.NS","RITES.NS","RADICO.NS","RVNL.NS","RAILTEL.NS","RAINBOW.NS","RKFORGE.NS","RCF.NS","RTNINDIA.NS",
    "RAYMONDLSL.NS","RAYMOND.NS","REDINGTON.NS","RELIANCE.NS","RPOWER.NS","ROUTE.NS","SBFC.NS","SBICARD.NS","SBILIFE.NS","SJVN.NS",
    "SKFINDIA.NS","SRF.NS","SAGILITY.NS","SAILIFE.NS","SAMMAANCAP.NS","MOTHERSON.NS","SAPPHIRE.NS","SARDAEN.NS","SAREGAMA.NS","SCHAEFFLER.NS",
    "SCHNEIDER.NS","SCI.NS","SHREECEM.NS","RENUKA.NS","SHRIRAMFIN.NS","SHYAMMETL.NS","SIEMENS.NS","SIGNATURE.NS","SOBHA.NS","SOLARINDS.NS",
    "SONACOMS.NS","SONATSOFTW.NS","STARHEALTH.NS","SBIN.NS","SAIL.NS","SWSOLAR.NS","SUMICHEM.NS","SUNPHARMA.NS","SUNTV.NS","SUNDARMFIN.NS",
    "SUNDRMFAST.NS","SUPREMEIND.NS","SUZLON.NS","SWANENERGY.NS","SWIGGY.NS","SYNGENE.NS","SYRMA.NS","TBOTEK.NS","TVSMOTOR.NS","TANLA.NS",
    "TATACHEM.NS","TATACOMM.NS","TCS.NS","TATACONSUM.NS","TATAELXSI.NS","TATAINVEST.NS","TATAMOTORS.NS","TATAPOWER.NS","TATASTEEL.NS","TATATECH.NS",
    "TTML.NS","TECHM.NS","TECHNOE.NS","TEJASNET.NS","NIACL.NS","RAMCOCEM.NS","THERMAX.NS","TIMKEN.NS","TITAGARH.NS","TITAN.NS",
    "TORNTPHARM.NS","TORNTPOWER.NS","TARIL.NS","TRENT.NS","TRIDENT.NS","TRIVENI.NS","TRITURBINE.NS","TIINDIA.NS","UCOBANK.NS","UNOMINDA.NS",
    "UPL.NS","UTIAMC.NS","ULTRACEMCO.NS","UNIONBANK.NS","UBL.NS","UNITDSPR.NS","USHAMART.NS","VGUARD.NS","DBREALTY.NS","VTL.NS",
    "VBL.NS","MANYAVAR.NS","VEDL.NS","VIJAYA.NS","VMM.NS","IDEA.NS","VOLTAS.NS","WAAREEENER.NS","WELCORP.NS","WELSPUNLIV.NS",
    "WESTLIFE.NS","WHIRLPOOL.NS","WIPRO.NS","WOCKPHARMA.NS","YESBANK.NS","ZFCVINDIA.NS","ZEEL.NS","ZENTEC.NS","ZENSARTECH.NS","ZYDUSLIFE.NS",
    "ECLERX.NS",
]

# ---------------- TIMEFRAME CONFIG ----------------
TIMEFRAME_CONFIG = {
    "Hourly": {
        "period": "60d",
        "interval": "1h",
        "sma_windows": (20, 50),
        "support_window": 30,
        "zz_pct": 0.005,   # 0.5%
        "zz_min_bars": 6,
        "horizon": 7,      # ~1 trading day on 1h chart
        "buy_thr": 0.01,
        "sell_thr": -0.01,
        "min_rows": 250,
    },
    "Daily": {
        "period": "3y",
        "interval": "1d",
        "sma_windows": (20, 50, 200),
        "support_window": 30,
        "zz_pct": 0.05,    # 5%
        "zz_min_bars": 5,
        "horizon": 60,
        "buy_thr": 0.03,
        "sell_thr": -0.03,
        "min_rows": 250,
    },
    "Weekly": {
        "period": "5y",
        "interval": "1wk",
        "sma_windows": (20, 50, 200),
        "support_window": 30,
        "zz_pct": 0.05,    # 5%
        "zz_min_bars": 5,
        "horizon": 8,
        "buy_thr": 0.05,
        "sell_thr": -0.05,
        "min_rows": 150,
    },
}

# ---------------- UTIL ----------------
class _TQDM:
    def __init__(self, total, desc=""):
        self.pb = st.progress(0, text=desc)
        self.total = max(total, 1)
        self.i = 0
    def update(self):
        self.i += 1
        self.pb.progress(min(self.i / self.total, 1.0), text=f"{self.i}/{self.total}")
    def close(self):
        self.pb.empty()

def stqdm(iterable, total=None, desc=""):
    if total is None:
        try:
            total = len(iterable)
        except Exception:
            total = 100
    bar = _TQDM(total=total, desc=desc)
    for x in iterable:
        yield x
        bar.update()
    bar.close()

def tradingview_link(ticker: str) -> str:
    return f"https://in.tradingview.com/chart/?symbol=NSE%3A{ticker.replace('.NS','')}"

@st.cache_data(show_spinner=False)
def load_history_for_ticker(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=True)
        return df
    except Exception:
        return pd.DataFrame()

# ---------------- ELLIOTT WAVE (ZigZag + Heuristics) ----------------
def zigzag_pivots(close: pd.Series, pct=0.05, min_bars=5):
    if close.isna().all() or len(close) < max(50, min_bars*4):
        return pd.DataFrame(columns=["idx", "price", "type"])

    c = close.values.astype(float)
    idxs = close.index

    piv = []
    last_piv_i = 0
    last_piv_p = c[0]
    trend = None
    last_extreme_i = 0
    last_extreme_p = c[0]

    for i in range(1, len(c)):
        if trend in (None, 'up'):
            if c[i] > last_extreme_p:
                last_extreme_p = c[i]; last_extreme_i = i
        if trend in (None, 'down'):
            if c[i] < last_extreme_p:
                last_extreme_p = c[i]; last_extreme_i = i

        if trend in (None, 'up'):
            dd = (c[i] - last_extreme_p) / last_extreme_p if last_extreme_p != 0 else 0
            if dd <= -pct and (i - last_piv_i) >= min_bars:
                piv.append((idxs[last_extreme_i], float(last_extreme_p), 'H'))
                last_piv_i = last_extreme_i; last_piv_p = last_extreme_p
                trend = 'down'
                last_extreme_i = i; last_extreme_p = c[i]
        if trend in (None, 'down'):
            uu = (c[i] - last_extreme_p) / last_extreme_p if last_extreme_p != 0 else 0
            if uu >= pct and (i - last_piv_i) >= min_bars:
                piv.append((idxs[last_extreme_i], float(last_extreme_p), 'L'))
                last_piv_i = last_extreme_i; last_piv_p = last_extreme_p
                trend = 'up'
                last_extreme_i = i; last_extreme_p = c[i]

    if len(piv) >= 2:
        cleaned = [piv[0]]
        for i in range(1, len(piv)):
            t_i = piv[i][2]
            t_prev = cleaned[-1][2]
            if t_i == t_prev:
                prev = cleaned[-1]
                if t_i == 'H':
                    better = piv[i][1] > prev[1]
                else:
                    better = piv[i][1] < prev[1]
                if better:
                    cleaned[-1] = piv[i]
            else:
                cleaned.append(piv[i])
        piv = cleaned

    if not piv:
        return pd.DataFrame(columns=["idx", "price", "type"])
    idx, price, typ = zip(*piv)
    return pd.DataFrame({"idx": list(idx), "price": list(price), "type": list(typ)})

def fib_okay(a, b, ratio, tol=0.18):
    if b == 0 or np.isnan(a) or np.isnan(b):
        return False
    return abs((a / b) - ratio) <= tol * ratio

def elliott_phase_from_pivots(pivots: pd.DataFrame):
    out = {"phase": "Unknown", "wave_no": 0, "bullish": False, "bearish": False}
    if pivots.empty:
        return out

    if len(pivots) >= 5:
        p5 = pivots.iloc[-5:].reset_index(drop=True)
        alt = all(p5.loc[i, "type"] != p5.loc[i-1, "type"] for i in range(1, 5))
        if alt:
            prices = p5["price"].values
            types = p5["type"].values
            up_pattern = (types.tolist() == ['L','H','L','H','L'])
            down_pattern = (types.tolist() == ['H','L','H','L','H'])
            if up_pattern:
                hh_ok = prices[3] > prices[1]
                hl_ok = prices[4] > prices[2]
                w1 = prices[1] - prices[0]
                w2 = prices[1] - prices[2]
                w3 = prices[3] - prices[2]
                w4 = prices[3] - prices[4]
                fib2 = fib_okay(w2, w1, 0.382) or fib_okay(w2, w1, 0.5) or fib_okay(w2, w1, 0.618)
                fib4 = fib_okay(w4, w3, 0.382) or fib_okay(w4, w3, 0.5) or fib_okay(w4, w3, 0.618)
                if hh_ok and hl_ok and (fib2 or fib4):
                    out.update({"phase": "ImpulseUp", "wave_no": 5, "bullish": True})
                    return out
            if down_pattern:
                ll_ok = prices[3] < prices[1]
                lh_ok = prices[4] < prices[2]
                w1 = prices[0] - prices[1]
                w2 = prices[2] - prices[1]
                w3 = prices[2] - prices[3]
                w4 = prices[4] - prices[3]
                fib2 = fib_okay(w2, w1, 0.382) or fib_okay(w2, w1, 0.5) or fib_okay(w2, w1, 0.618)
                fib4 = fib_okay(w4, w3, 0.382) or fib_okay(w4, w3, 0.5) or fib_okay(w4, w3, 0.618)
                if ll_ok and lh_ok and (fib2 or fib4):
                    out.update({"phase": "ImpulseDown", "wave_no": 5, "bearish": True})
                    return out

    if len(pivots) >= 3:
        p3 = pivots.iloc[-3:].reset_index(drop=True)
        alt3 = all(p3.loc[i, "type"] != p3.loc[i-1, "type"] for i in range(1, 3))
        if alt3:
            t = p3["type"].tolist()
            if t == ['L','H','L']:
                out.update({"phase": "CorrectionUp", "wave_no": 3, "bullish": True})
            elif t == ['H','L','H']:
                out.update({"phase": "CorrectionDown", "wave_no": 3, "bearish": True})
    return out

def add_elliott_features_core(df_close: pd.Series, pct=0.05, min_bars=5):
    piv = zigzag_pivots(df_close, pct=pct, min_bars=min_bars)
    phase = elliott_phase_from_pivots(piv)
    return phase, piv

# ---------------- FEATURE ENGINEERING ----------------
def compute_features(df, sma_windows=(20, 50, 200), support_window=30, zz_pct=0.05, zz_min_bars=5):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if "Close" not in df.columns or df["Close"].dropna().empty:
        return pd.DataFrame()

    df = df.copy()

    try:
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    except Exception:
        df["RSI"] = np.nan

    for win in sma_windows:
        df[f"SMA{win}"] = df["Close"].rolling(window=win, min_periods=1).mean()

    df["Support"] = df["Close"].rolling(window=support_window, min_periods=1).min()

    df["RSI_Direction"] = df["RSI"].diff(5)
    df["Price_Direction"] = df["Close"].diff(5)
    df["Bullish_Div"] = (df["RSI_Direction"] > 0) & (df["Price_Direction"] < 0)
    df["Bearish_Div"] = (df["RSI_Direction"] < 0) & (df["Price_Direction"] > 0)

    for w in (1, 3, 5, 10):
        df[f"Ret_{w}"] = df["Close"].pct_change(w)

    for win in sma_windows:
        df[f"Dist_SMA{win}"] = (df["Close"] - df[f"SMA{win}"]) / df[f"SMA{win}"]

    for col in ["RSI"] + [f"SMA{w}" for w in sma_windows]:
        df[f"{col}_slope"] = df[col].diff()

    try:
        phase, piv = add_elliott_features_core(df["Close"], pct=zz_pct, min_bars=zz_min_bars)
        phase_map = {
            "ImpulseUp": 1, "ImpulseDown": -1,
            "CorrectionUp": 2, "CorrectionDown": -2,
            "Unknown": 0
        }
        df["Elliott_Phase_Code"] = phase_map.get(phase["phase"], 0)
        df["Elliott_Wave_No"] = int(phase.get("wave_no", 0))
        df["Elliott_Bullish"] = bool(phase.get("bullish", False))
        df["Elliott_Bearish"] = bool(phase.get("bearish", False))
        df["Elliott_Bullish_Int"] = df["Elliott_Bullish"].astype(int)
        df["Elliott_Bearish_Int"] = df["Elliott_Bearish"].astype(int)
    except Exception:
        df["Elliott_Phase_Code"] = 0
        df["Elliott_Wave_No"] = 0
        df["Elliott_Bullish"] = False
        df["Elliott_Bearish"] = False
        df["Elliott_Bullish_Int"] = 0
        df["Elliott_Bearish_Int"] = 0

    return df

# ---------------- LABELS ----------------
def label_from_future_returns(df, horizon=7, buy_thr=0.01, sell_thr=-0.01):
    fut_ret = df["Close"].shift(-horizon) / df["Close"] - 1.0
    label = pd.Series(0, index=df.index, dtype=int)
    label[fut_ret >= buy_thr] = 1
    label[fut_ret <= sell_thr] = -1
    return label

# ---------------- ML HELPERS ----------------
def train_rf_classifier(X, y, random_state=42):
    if X.empty or y.empty:
        return None, None, None
    stratify_opt = y if len(np.unique(y)) > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, stratify=stratify_opt, random_state=random_state
        )
    except Exception:
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=False)
    return clf, acc, report

def latest_feature_row_for_ticker(ticker, cfg, feature_cols):
    hist = load_history_for_ticker(ticker, cfg["period"], cfg["interval"])
    if hist is None or hist.empty:
        return None
    feat = compute_features(
        hist,
        sma_windows=cfg["sma_windows"],
        support_window=cfg["support_window"],
        zz_pct=cfg["zz_pct"],
        zz_min_bars=cfg["zz_min_bars"],
    ).dropna()
    if feat.empty:
        return None
    use = feat.select_dtypes(include=[np.number])
    row = use.iloc[-1:].copy()
    for m in [c for c in feature_cols if c not in row.columns]:
        row[m] = 0.0
    row = row[feature_cols]
    return row

def build_and_score_for_timeframe(timeframe_name, tickers):
    cfg = TIMEFRAME_CONFIG[timeframe_name]
    X_list, y_list, meta_list = [], [], []
    feature_cols = None

    for t in stqdm(tickers, desc=f"Preparing ML data ({timeframe_name})"):
        hist = load_history_for_ticker(t, cfg["period"], cfg["interval"])
        if hist is None or hist.empty or len(hist) < cfg["min_rows"]:
            continue

        feat = compute_features(
            hist,
            sma_windows=cfg["sma_windows"],
            support_window=cfg["support_window"],
            zz_pct=cfg["zz_pct"],
            zz_min_bars=cfg["zz_min_bars"],
        )
        if feat.empty:
            continue

        y = label_from_future_returns(
            feat,
            horizon=cfg["horizon"],
            buy_thr=cfg["buy_thr"],
            sell_thr=cfg["sell_thr"],
        )

        data = feat.join(y.rename("Label")).dropna()
        if data.empty:
            continue

        drop_cols = set(["Label", "Support", "Bullish_Div", "Bearish_Div"])
        use = data.select_dtypes(include=[np.number]).drop(columns=list(drop_cols & set(data.columns)), errors="ignore")

        if feature_cols is None:
            feature_cols = list(use.columns)

        X_list.append(use[feature_cols])
        y_list.append(data["Label"])
        meta_list.append(pd.Series([t] * len(use), index=use.index, name="Ticker"))

    if not X_list:
        return None, None

    X = pd.concat(X_list, axis=0)
    y = pd.concat(y_list, axis=0)

    clf, acc, report = train_rf_classifier(X, y)
    if clf is None:
        return None, None

    # Score latest snapshot for each ticker
    rows = []
    for t in stqdm(tickers, desc=f"Scoring ({timeframe_name})", total=len(tickers)):
        row = latest_feature_row_for_ticker(t, cfg, feature_cols)
        if row is None:
            continue
        proba = clf.predict_proba(row)[0] if hasattr(clf, "predict_proba") else None
        pred = clf.predict(row)[0]

        rows.append({
            "Ticker": t,
            "Timeframe": timeframe_name,
            "ML_Pred": {1: "BUY", 0: "HOLD", -1: "SELL"}.get(int(pred), "HOLD"),
            "Prob_Buy": float(proba[list(clf.classes_).index(1)]) if proba is not None and 1 in clf.classes_ else np.nan,
            "Prob_Hold": float(proba[list(clf.classes_).index(0)]) if proba is not None and 0 in clf.classes_ else np.nan,
            "Prob_Sell": float(proba[list(clf.classes_).index(-1)]) if proba is not None and -1 in clf.classes_ else np.nan,
        })

    ml_df = pd.DataFrame(rows)
    return ml_df, (acc, report)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")

    timeframes_selected = st.multiselect(
        "Select timeframe(s)",
        ["Hourly", "Daily", "Weekly"],
        default=["Daily"]
    )

    if not timeframes_selected:
        st.warning("Select at least one timeframe.")
    
    select_all = st.checkbox("Select all stocks", value=True)
    default_list = NIFTY500_TICKERS if select_all else NIFTY500_TICKERS[:25]
    selected_tickers = st.multiselect(
        "Select stocks",
        NIFTY500_TICKERS,
        default=default_list
    )

    prob_cutoff = st.slider(
        "Min BUY/SELL probability (for multi-timeframe consensus)",
        0.50, 0.99, 0.80, step=0.01
    )

    run_analysis = st.button("Run Analysis")

# ---------------- MAIN ----------------
if run_analysis:
    if not SKLEARN_OK:
        st.error("scikit-learn not available. Install with: pip install scikit-learn")
        st.stop()

    if not timeframes_selected:
        st.error("Please select at least one timeframe.")
        st.stop()

    if not selected_tickers:
        st.error("Please select at least one stock.")
        st.stop()

    ml_results = {}
    metrics = {}

    for tf in timeframes_selected:
        with st.spinner(f"Building ML model for {tf} timeframe..."):
            ml_df, (acc, report) = build_and_score_for_timeframe(tf, selected_tickers)
            if ml_df is None or ml_df.empty:
                st.warning(f"No ML data for {tf} timeframe (maybe not enough history).")
                continue
            ml_df["TradingView"] = ml_df["Ticker"].apply(tradingview_link)
            ml_results[tf] = ml_df
            metrics[tf] = (acc, report)

    if not ml_results:
        st.error("No ML results available for the chosen settings.")
        st.stop()

    # If only one timeframe: show that table directly
    if len(ml_results) == 1:
        tf = list(ml_results.keys())[0]
        st.subheader(f"🤖 ML Signals — {tf} timeframe")
        acc, report = metrics[tf]
        st.caption(f"Validation accuracy (holdout): **{acc:.3f}**")
        with st.expander("Classification report"):
            st.text(report)

        ml_df = ml_results[tf].copy()
        ml_df = ml_df.sort_values(["ML_Pred", "Prob_Buy"], ascending=[True, False])

        st.dataframe(
            ml_df,
            use_container_width=True,
            column_config={
                "TradingView": st.column_config.LinkColumn(
                    "TradingView",
                    display_text="📈 Chart"
                ),
                "Prob_Buy": st.column_config.ProgressColumn("Prob_Buy", min_value=0.0, max_value=1.0),
                "Prob_Hold": st.column_config.ProgressColumn("Prob_Hold", min_value=0.0, max_value=1.0),
                "Prob_Sell": st.column_config.ProgressColumn("Prob_Sell", min_value=0.0, max_value=1.0),
            }
        )

        csv_df = ml_df.drop(columns=["TradingView"], errors="ignore")
        st.download_button(
            label="📥 Download ML Signals as CSV",
            data=csv_df.to_csv(index=False).encode("utf-8"),
            file_name=f"ml_signals_{tf.lower()}.csv",
            mime="text/csv",
        )
    else:
        # -------- MULTI-TIMEFRAME CONSENSUS --------
        st.subheader("🤝 Multi-Timeframe Consensus (common tickers)")

        # 1️⃣ Debug: per-timeframe strong signals (BUY/SELL & prob >= cutoff)
        st.markdown("### 📊 Per-timeframe strong signals before consensus")

        strong_sets = {}
        for tf, df_tf in ml_results.items():
            df_tf = df_tf.copy()
            df_tf["Max_BuySell"] = df_tf[["Prob_Buy", "Prob_Sell"]].max(axis=1)
            df_tf["Is_Strong"] = df_tf["ML_Pred"].isin(["BUY", "SELL"]) & (df_tf["Max_BuySell"] >= prob_cutoff)

            strong_df = df_tf[df_tf["Is_Strong"]].copy()
            strong_sets[tf] = set(strong_df["Ticker"])

            st.write(f"**{tf}**: {len(strong_df)} stocks with BUY/SELL and prob ≥ {prob_cutoff*100:.0f}%")
            if not strong_df.empty:
                st.dataframe(
                    strong_df[["Ticker", "ML_Pred", "Prob_Buy", "Prob_Sell", "Prob_Hold"]],
                    use_container_width=True,
                )

        if not strong_sets:
            st.warning("No strong signals in any timeframe.")
            st.stop()

        # 2️⃣ Consensus tickers: intersection of strong sets
        if len(strong_sets) == 1:
            consensus_tickers = list(next(iter(strong_sets.values())))
        else:
            consensus_tickers = set.intersection(*strong_sets.values())

        if not consensus_tickers:
            st.info(f"No tickers satisfy consensus BUY/SELL with ≥ {prob_cutoff*100:.0f}% probability in all selected timeframes.")
            st.stop()

        # 3️⃣ Build final consensus table (average confidence across timeframes)
        rows = []
        for ticker in consensus_tickers:
            row = {"Ticker": ticker}
            max_probs = []

            for tf in timeframes_selected:
                df_tf = ml_results[tf]
                r = df_tf[df_tf["Ticker"] == ticker]
                if r.empty:
                    continue
                r = r.iloc[0]

                pred = r["ML_Pred"]
                pb = r["Prob_Buy"]
                ps = r["Prob_Sell"]
                ph = r["Prob_Hold"]
                max_bs = max(pb, ps)

                row[f"Pred_{tf}"] = pred
                row[f"Prob_Buy_{tf}"] = pb
                row[f"Prob_Sell_{tf}"] = ps
                row[f"Prob_Hold_{tf}"] = ph
                row[f"Max_Prob_{tf}"] = max_bs
                max_probs.append(max_bs)

            # overall average confidence
            row["Avg_Max_Prob"] = np.mean(max_probs) if max_probs else np.nan
            rows.append(row)

        consensus_df = pd.DataFrame(rows)

        # Sort by highest average confidence
        consensus_df = consensus_df.sort_values("Avg_Max_Prob", ascending=False)

        # Add TradingView link
        consensus_df["TradingView"] = consensus_df["Ticker"].apply(tradingview_link)

        st.markdown("### ✅ Final consensus list")
        st.dataframe(
            consensus_df,
            use_container_width=True,
            column_config={
                "TradingView": st.column_config.LinkColumn(
                    "TradingView",
                    display_text="📈 Chart"
                ),
                "Avg_Max_Prob": st.column_config.ProgressColumn(
                    "Avg Max Prob (across TFs)", min_value=0.0, max_value=1.0
                ),
                **{
                    col: st.column_config.ProgressColumn(col, min_value=0.0, max_value=1.0)
                    for col in consensus_df.columns
                    if col.startswith("Prob_")
                },
            }
        )

        csv_df = consensus_df.drop(columns=["TradingView"], errors="ignore")
        st.download_button(
            label="📥 Download Consensus ML Signals as CSV",
            data=csv_df.to_csv(index=False).encode("utf-8"),
            file_name="ml_signals_consensus_multi_timeframe.csv",
            mime="text/csv",
        )          
