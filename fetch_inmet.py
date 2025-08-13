import os, re, io, zipfile, argparse, unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Iterable, Tuple, Dict

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

BASE_URL = "https://portal.inmet.gov.br/dadoshistoricos"

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    return text.strip("_").lower()

TARGET_SLUG = slugify("SAO LUIZ DO PARAITINGA")

def get(session: requests.Session, url: str, **kw) -> requests.Response:
    r = session.get(url, timeout=60, **kw)
    r.raise_for_status()
    return r

def find_year_links(html: str) -> Dict[int, str]:
    soup = BeautifulSoup(html, "html.parser")  # parser mais tolerante
    links = {}
    for a in soup.find_all("a"):
        text = (a.get_text() or "").upper()
        m = re.search(r"ANO\s+(\d{4}).*AUTOM", text)
        href = a.get("href")
        if m and href:
            y = int(m.group(1))
            if not href.startswith("http"):
                href = "https://portal.inmet.gov.br" + href
            links[y] = href
    return links

def find_zip_links(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")  # parser mais tolerante
    out = []
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if href.lower().endswith(".zip"):
            if not href.startswith("http"):
                href = "https://portal.inmet.gov.br" + href
            out.append(href)
    return out

def iter_csv_from_zip(content: bytes) -> Iterable[Tuple[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for info in zf.infolist():
            if info.filename.lower().endswith(".csv"):
                with zf.open(info) as f:
                    yield info.filename, f.read()

def try_read_csv(bytes_content: bytes) -> pd.DataFrame:
    for enc in ["latin-1","utf-8-sig","utf-8"]:
        try:
            return pd.read_csv(io.BytesIO(bytes_content), sep=";", encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(io.BytesIO(bytes_content), encoding="latin-1", low_memory=False)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    def norm(c):
        c2 = unicodedata.normalize("NFKD", str(c)).encode("ascii","ignore").decode("ascii").upper()
        c2 = re.sub(r"[^A-Z0-9]+","_", c2).strip("_")
        return c2
    df = df.rename(columns={c: norm(c) for c in df.columns})

    data_cols = [c for c in df.columns if c in ("DATA","DATA_MEDICAO","DT_MEDICAO","DT_MED")]
    hora_cols = [c for c in df.columns if c.startswith("HORA")]
    if "DATA" not in df.columns and data_cols:
        df["DATA"] = df[data_cols[0]]
    if hora_cols:
        try:
            df["DATA"] = pd.to_datetime(df["DATA"].astype(str) + " " + df[hora_cols[0]].astype(str), errors="coerce", dayfirst=True)
        except Exception:
            df["DATA"] = pd.to_datetime(df.get("DATA"), errors="coerce", dayfirst=True)
    else:
        df["DATA"] = pd.to_datetime(df.get("DATA"), errors="coerce", dayfirst=True)

    out = pd.DataFrame()
    out["DATA"] = df["DATA"]

    def pick(cols):
        return df[cols[0]] if cols else pd.Series([None]*len(df))

    cand_temp = [c for c in df.columns if "TEMPERATURA" in c and ("MEDIA" in c or "AR" in c or "BULBO" in c)]
    cand_tmax = [c for c in df.columns if "TEMPERATURA_MAX" in c]
    cand_tmin = [c for c in df.columns if "TEMPERATURA_MIN" in c]
    cand_umid = [c for c in df.columns if "UMIDADE" in c and ("REL" in c or "RELATIVA" in c)]
    cand_prec = [c for c in df.columns if "PRECIP" in c]
    cand_vento = [c for c in df.columns if "VENTO" in c and ("VELOC" in c or "VEL" in c)]
    cand_press = [c for c in df.columns if "PRESSAO" in c]

    out["TEMPERATURA_MEDIA"] = pd.to_numeric(pick([c for c in cand_temp if "MEDIA" in c] or cand_temp), errors="coerce")
    out["TEMPERATURA_MAXIMA"] = pd.to_numeric(pick(cand_tmax), errors="coerce")
    out["TEMPERATURA_MINIMA"] = pd.to_numeric(pick(cand_tmin), errors="coerce")
    out["UMIDADE_RELATIVA"] = pd.to_numeric(pick(cand_umid), errors="coerce")
    out["PRECIPITACAO"] = pd.to_numeric(pick(cand_prec), errors="coerce")
    out["VELOCIDADE_VENTO"] = pd.to_numeric(pick(cand_vento), errors="coerce")
    out["PRESSAO_ATMOSFERICA"] = pd.to_numeric(pick(cand_press), errors="coerce")

    if out["TEMPERATURA_MEDIA"].isna().all():
        if not out["TEMPERATURA_MAXIMA"].isna().all() and not out["TEMPERATURA_MINIMA"].isna().all():
            out["TEMPERATURA_MEDIA"] = (out["TEMPERATURA_MAXIMA"] + out["TEMPERATURA_MINIMA"]) / 2.0

    return out

def is_target_station(filename: str) -> bool:
    return TARGET_SLUG in slugify(filename)

def download_and_extract_for_year(session: requests.Session, year: int, out_raw: Path):
    print(f"Processando ano {year}...")
    res = get(session, BASE_URL)
    year_links = find_year_links(res.text)
    if year not in year_links:
        print(f"Nenhum link encontrado para o ano {year}.")
        return []

    year_url = year_links[year]
    res_y = get(session, year_url)

    # --- Tratamento para link direto ZIP ---
    content_type = res_y.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type:
        print(f"[{year}] Link direto para arquivo detectado (Content-Type: {content_type})")
        zip_links = [year_url]
    else:
        try:
            zip_links = find_zip_links(res_y.text)
        except Exception as e:
            print(f"[{year}] Erro ao parsear HTML: {e}")
            return []

    out_dfs = []
    for zurl in tqdm(zip_links, desc=f"{year} - zips"):
        try:
            r = get(session, zurl, stream=True)
            content = r.content
            out_raw.mkdir(parents=True, exist_ok=True)
            with open(out_raw / f"{year}_{os.path.basename(zurl)}", "wb") as f:
                f.write(content)
            for fname, bytes_csv in iter_csv_from_zip(content):
                if is_target_station(fname):
                    df = try_read_csv(bytes_csv)
                    out_dfs.append(normalize_columns(df))
        except Exception as e:
            print(f"[{year}] Falha no arquivo {zurl}: {e}")
    return out_dfs

def main():
    parser = argparse.ArgumentParser(description="Baixa dados do INMET para São Luiz do Paraitinga e combina em um CSV.")
    parser.add_argument("--years", default="all", help="all | 2000-2025 | 2018,2019,...")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--combined", default="data/inmet_data_sao_luiz_do_paraitinga_combined.csv")
    args = parser.parse_args()

    if args.years == "all":
        years = list(range(2000, datetime.now().year + 1))
    elif "-" in args.years:
        a, b = args.years.split("-")
        years = list(range(int(a), int(b) + 1))
    else:
        years = [int(x) for x in re.split(r"[,\s]+", args.years.strip()) if x]

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; INMETFetcher/1.0)"})
    all_dfs = []
    for y in years:
        dfs = download_and_extract_for_year(session, y, Path(args.raw_dir) / str(y))
        all_dfs.extend(dfs)

    if not all_dfs:
        print("Nenhum dado encontrado para a estação alvo.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["DATA"]).sort_values("DATA").drop_duplicates(subset=["DATA"])

    out_path = Path(args.combined)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print(f"✔ Arquivo combinado salvo em: {out_path} (linhas: {len(df_all)})")

if __name__ == "__main__":
    main()

