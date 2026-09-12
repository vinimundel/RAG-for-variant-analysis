import os
import hashlib
import json
import requests, re
from pathlib import Path
from typing import Dict, List, Optional
import urllib.parse

class PDFDownloader:
    """Automates fetching, downloading, and local caching of full-text scientific paper PDFs via PMC, Unpaywall, OpenAlex, Elsevier, and Semantic Scholar APIs."""

    def __init__(self, data_root: Optional[Path] = None):
        if data_root is None:
            self.data_root = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_root = Path(data_root)

        # Optional credentials come from the invoking process; Pixi is the sole
        # dependency environment and no project-local .env file is loaded.
        self.elsevier_api_key = os.environ.get("ELSEVIER_API_KEY")
        if self.elsevier_api_key in ["sua_chave_elsevier_aqui", "your_elsevier_api_key_here"]:
            self.elsevier_api_key = None

    def resolve_pubmed_metadata(self, pmid: str) -> Dict[str, Optional[str]]:
        """Resolves PMCID, DOI, and OpenAccess status for a PMID via Europe PMC and NCBI ID Converter APIs."""
        if not pmid or not str(pmid).isdigit():
            return {"pmcid": None, "doi": None, "isOpenAccess": "N", "ft_pdf_urls": []}

        pmcid = None
        doi = None
        is_oa = "N"
        ft_urls = []

        # 1. Europe PMC API
        try:
            url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                   f"?query=EXT_ID:{pmid}%20SRC:MED&format=json&resultType=core")
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                results = res.json().get("resultList", {}).get("result", [])
                if results:
                    item = results[0]
                    pmcid = item.get("pmcid")
                    doi = item.get("doi")
                    is_oa = item.get("isOpenAccess", "N")
                    url_list = item.get("fullTextUrlList", {}).get("fullTextUrl", [])
                    for u in url_list:
                        if u.get("documentStyle") == "pdf":
                            ft_urls.append(u.get("url"))
        except Exception:
            pass

        # Try the NCBI ID Converter if PMCID or DOI metadata is incomplete.
        if not pmcid or not doi:
            try:
                conv_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
                res_ncbi = requests.get(conv_url, timeout=10)
                if res_ncbi.status_code == 200:
                    rec = res_ncbi.json().get("records", [{}])[0]
                    if not pmcid:
                        pmcid = rec.get("pmcid")
                    if not doi:
                        doi = rec.get("doi")
            except Exception:
                pass

        return {
            "pmcid": pmcid,
            "doi": doi,
            "isOpenAccess": is_oa,
            "ft_pdf_urls": ft_urls
        }

    def get_pdf_url_from_openalex(self, pmid: str = None, doi: str = None) -> Optional[str]:
        """Queries OpenAlex API for open access PDF link."""
        query_path = None
        if pmid and str(pmid).isdigit():
            query_path = f"pmid:{pmid}"
        elif doi and doi != "N/A":
            query_path = f"https://doi.org/{doi}"

        if not query_path:
            return None

        try:
            url = f"https://api.openalex.org/works/{query_path}"
            res = requests.get(url, headers={"User-Agent": "mailto:researcher@example.org"}, timeout=10)
            if res.status_code == 200:
                data = res.json()
                best_oa = data.get("best_oa_location") or {}
                pdf_u = best_oa.get("pdf_url")
                if pdf_u:
                    return pdf_u
                for loc in data.get("locations", []):
                    if loc.get("pdf_url"):
                        return loc.get("pdf_url")
        except Exception:
            pass
        return None

    def fetch_pdf_via_elsevier_api(self, doi: str, output_path: Path) -> bool:
        """Downloads full text PDF directly via official Elsevier ScienceDirect API using API Key."""
        if not doi or doi == "N/A" or not self.elsevier_api_key:
            return False

        try:
            url = f"https://api.elsevier.com/content/article/doi/{doi}"
            headers = {
                "X-ELS-APIKey": self.elsevier_api_key,
                "Accept": "application/pdf"
            }
            res = requests.get(url, headers=headers, stream=True, timeout=30)
            if res.status_code == 200:
                content_start = res.raw.read(10)
                if b"%PDF" in content_start or b"PDF" in content_start:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(content_start)
                        for chunk in res.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return True
        except Exception as e:
            print(f"Elsevier API fetch error for DOI {doi}: {e}")
        return False

    def get_pdf_url_from_unpaywall(self, doi: str) -> Optional[str]:
        """Queries Unpaywall API for Open Access PDF URL using paper DOI."""
        if not doi or doi == "N/A":
            return None

        try:
            url = f"https://api.unpaywall.org/v2/{doi}?email=researcher@example.org"
            res = requests.get(url, timeout=12)
            if res.status_code == 200:
                data = res.json()
                best_loc = data.get("best_oa_location") or {}
                pdf_url = best_loc.get("url_for_pdf")
                if pdf_url:
                    return pdf_url
                for loc in data.get("oa_locations", []):
                    u = loc.get("url_for_pdf") or loc.get("url")
                    if u and ("pdf" in u.lower() or "pmc" in u.lower()):
                        return u
        except Exception:
            pass
        return None

    def get_pdf_url_from_semantic_scholar(self, pmid: str) -> Optional[str]:
        """Queries Semantic Scholar API for Open Access PDF link."""
        if not pmid or pmid == "N/A" or not str(pmid).isdigit():
            return None

        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/PMID:{pmid}?fields=openAccessPdf"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                oa_pdf = res.json().get("openAccessPdf") or {}
                url = oa_pdf.get("url")
                if url and "pdf" in url.lower():
                    return url
        except Exception:
            pass
        return None

    def download_pdf_file(self, pdf_url: str, output_path: Path) -> bool:
        """Downloads PDF file from URL and validates header."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        }
        try:
            res = requests.get(pdf_url, headers=headers, stream=True, timeout=25, allow_redirects=True)
            if res.status_code == 200:
                content_start = res.raw.read(10)
                if b"%PDF" in content_start or b"PDF" in content_start:
                    with open(output_path, "wb") as f:
                        f.write(content_start)
                        for chunk in res.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return True
        except Exception:
            pass
        return False

    def process_publication_pdfs(self, publications_list: List[Dict], gene_symbol: str) -> List[Dict]:
        """Fetches, caches, and updates PDF status for a list of publication dictionaries."""
        pdf_dir = self.data_root / "input" / gene_symbol / "evidence" / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        updated_pubs = []
        for pub in publications_list:
            pdb_id = str(pub.get("pdb_id", "PDB")).upper()
            pmid = str(pub.get("pubmed_id", "N/A"))
            doi = str(pub.get("doi", "N/A"))

            safe_pmid = pmid if pmid != "N/A" else "NO_PMID"
            filename = f"{pdb_id}_PMID{safe_pmid}.pdf"
            pdf_path = pdf_dir / filename

            # Check Local Cache
            if pdf_path.exists() and pdf_path.stat().st_size > 1024:
                pub["pdf_status"] = "Cached (Local File)"
                pub["pdf_path"] = str(pdf_path)
                pub["pdf_filename"] = filename
                updated_pubs.append(pub)
                continue

            # Resolvers Order: Elsevier API -> OpenAlex -> Unpaywall -> Semantic Scholar
            success = False
            if doi and self.elsevier_api_key:
                success = self.fetch_pdf_via_elsevier_api(doi, pdf_path)

            if not success:
                candidate_urls = []
                oa_url = self.get_pdf_url_from_openalex(pmid=pmid, doi=doi)
                if oa_url: candidate_urls.append(oa_url)

                unp_url = self.get_pdf_url_from_unpaywall(doi)
                if unp_url: candidate_urls.append(unp_url)

                ss_url = self.get_pdf_url_from_semantic_scholar(pmid)
                if ss_url: candidate_urls.append(ss_url)

                for u in candidate_urls:
                    if self.download_pdf_file(u, pdf_path):
                        success = True
                        break

            if success and pdf_path.exists():
                pub["pdf_status"] = "Downloaded (Open Access)"
                pub["pdf_path"] = str(pdf_path)
                pub["pdf_filename"] = filename
            else:
                pub["pdf_status"] = "Paywall / Direct Web Link"
                pub["pdf_path"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != "N/A" else (f"https://doi.org/{doi}" if doi != "N/A" else "N/A")
                pub["pdf_filename"] = "N/A"

            updated_pubs.append(pub)

        return updated_pubs

    def process_literature_csv(self, csv_path: Path, gene_symbol: str, pubmed_only: bool = True) -> Path:
        """Processes literature CSV file, fetches PDFs for available PubMed/PMC articles, and saves to input/<gene>/evidence/."""
        import pandas as pd

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Literature CSV not found at {csv_path}")

        df = pd.read_csv(csv_path)

        # Standardize columns
        title_col = None
        pubmed_col = None
        for col in df.columns:
            if col.lower() in ["title", "titulo", "article_title"]:
                title_col = col
            elif col.lower() in ["pubmed", "pmid", "pubmed_id"]:
                pubmed_col = col

        if not title_col:
            raise ValueError("CSV must contain a 'Title' column.")

        out_evidence_dir = self.data_root / "input" / gene_symbol / "evidence"
        pdf_dir = out_evidence_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total_rows = len(df)
        print(f"Processing {total_rows} literature items for gene {gene_symbol} (pubmed_only={pubmed_only})...")

        for idx, row in df.iterrows():
            title = str(row[title_col]).strip() if pd.notna(row[title_col]) else "Untitled"
            raw_pmid = str(row[pubmed_col]).strip() if (pubmed_col and pd.notna(row[pubmed_col])) else ""

            # Clean PMID (e.g. 15289881 or 15289881.0)
            if raw_pmid.endswith(".0"):
                raw_pmid = raw_pmid[:-2]
            pmid = raw_pmid if raw_pmid.isdigit() else ""

            if pubmed_only and not pmid:
                results.append({
                    "title": title,
                    "pubmed_id": "N/A",
                    "pmcid": "N/A",
                    "doi": "N/A",
                    "pdf_status": "Skipped (No PMID provided - Phase 1)",
                    "pdf_path": "N/A",
                    "pdf_filename": "N/A"
                })
                continue

            safe_id = f"PMID{pmid}" if pmid else f"ROW{idx+1}"
            filename = f"{safe_id}.pdf"
            pdf_path = pdf_dir / filename

            # Check cache
            if pdf_path.exists() and pdf_path.stat().st_size > 1024:
                print(f"[{idx+1}/{total_rows}] Cached: {filename}")
                results.append({
                    "title": title,
                    "pubmed_id": pmid if pmid else "N/A",
                    "pmcid": "Cached",
                    "doi": "Cached",
                    "pdf_status": "Cached (Local File)",
                    "pdf_path": str(pdf_path),
                    "pdf_filename": filename
                })
                continue

            # 1. Resolve metadata via Europe PMC & NCBI
            meta = self.resolve_pubmed_metadata(pmid) if pmid else {}
            pmcid = meta.get("pmcid")
            doi = meta.get("doi")
            ft_urls = meta.get("ft_pdf_urls", [])

            # First: Try direct Elsevier ScienceDirect API if API key is available
            success = False
            if doi and self.elsevier_api_key:
                success = self.fetch_pdf_via_elsevier_api(doi, pdf_path)
                if success:
                    print(f"[{idx+1}/{total_rows}] ✓ Downloaded via Elsevier API {filename} (PMID {pmid})")

            if not success:
                # Candidate URLs to try
                candidate_urls = []

                # Add OpenAlex link
                openalex_url = self.get_pdf_url_from_openalex(pmid=pmid, doi=doi)
                if openalex_url:
                    candidate_urls.append(openalex_url)

                # Add Unpaywall link (if DOI exists)
                if doi:
                    unp_url = self.get_pdf_url_from_unpaywall(doi)
                    if unp_url:
                        candidate_urls.append(unp_url)

                # Add Europe PMC direct FT links
                candidate_urls.extend(ft_urls)

                # Add Semantic Scholar link
                if pmid:
                    ss_url = self.get_pdf_url_from_semantic_scholar(pmid)
                    if ss_url:
                        candidate_urls.append(ss_url)

                for url in candidate_urls:
                    success = self.download_pdf_file(url, pdf_path)
                    if success:
                        break

            if success and pdf_path.exists():
                print(f"[{idx+1}/{total_rows}] ✓ Downloaded {filename} (PMID {pmid})")
                status = "Downloaded (Open Access)"
                filepath_str = str(pdf_path)
                fname_str = filename
            else:
                if pmid:
                    print(f"[{idx+1}/{total_rows}] x Paywall / Not Open Access (PMID {pmid})")
                    status = "Paywall / Direct Web Link"
                    filepath_str = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                else:
                    status = "No PMID / Pending Phase 2"
                    filepath_str = "N/A"
                fname_str = "N/A"

            results.append({
                "title": title,
                "pubmed_id": pmid if pmid else "N/A",
                "pmcid": pmcid if pmcid else "N/A",
                "doi": doi if doi else "N/A",
                "pdf_status": status,
                "pdf_path": filepath_str,
                "pdf_filename": fname_str
            })

        summary_df = pd.DataFrame(results)
        manifest_path = out_evidence_dir / f"{gene_symbol.lower()}_literature_evidence.csv"
        summary_df.to_csv(manifest_path, index=False)
        print(f"\nProcessing complete! Manifest saved to: {manifest_path}")
        print(f"Total downloaded: {len(summary_df[summary_df['pdf_status'].str.contains('Downloaded|Cached')])} / {len(summary_df)}")
        return manifest_path

    def process_literature_csv_concurrent(self, csv_path: Path, gene_symbol: str,
                                          max_workers: int = 8) -> Path:
        """Download unique PMID records concurrently with a complete status manifest."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import pandas as pd

        frame = pd.read_csv(csv_path)
        title_col = next((column for column in frame if column.lower() == "title"), None)
        pmid_col = next((column for column in frame if column.lower() in
                         {"pubmed", "pmid", "pubmed_id"}), None)
        if title_col is None or pmid_col is None:
            raise ValueError("CSV must contain Title and PubMed/PMID columns")
        evidence_dir = self.data_root / "input" / gene_symbol.upper() / "evidence"
        pdf_dir = evidence_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        def process(index_row):
            index, row = index_row
            raw = str(row[pmid_col]).removesuffix(".0") if pd.notna(row[pmid_col]) else ""
            title = str(row[title_col]).strip() if pd.notna(row[title_col]) else "Untitled"
            source_collection = str(row.get("source_collection", "literature_csv") or "literature_csv")
            base = {"source_row": int(index) + 1, "title": title,
                    "pubmed_id": raw if raw.isdigit() else "N/A",
                    "source_collection": source_collection}
            if not raw.isdigit():
                return {**base, "pmcid": "N/A", "doi": "N/A",
                        "pdf_status": "Skipped (No PMID)", "pdf_path": "N/A",
                        "pdf_filename": "N/A", "download_source": "N/A"}
            path, filename = pdf_dir / f"PMID{raw}.pdf", f"PMID{raw}.pdf"
            if path.exists() and path.stat().st_size > 1024:
                return {**base, "pmcid": "Cached", "doi": "Cached",
                        "pdf_status": "Cached (Local File)", "pdf_path": str(path),
                        "pdf_filename": filename, "download_source": "cache"}
            metadata = self.resolve_pubmed_metadata(raw)
            pmcid, doi = metadata.get("pmcid"), metadata.get("doi")
            candidates = [(url, "Europe PMC") for url in metadata.get("ft_pdf_urls", []) if url]
            if pmcid:
                candidates.append((f"https://europepmc.org/articles/{pmcid}/bin/{pmcid}.pdf",
                                   "Europe PMC PMCID"))
            source, success = None, False
            if doi and self.elsevier_api_key:
                success = self.fetch_pdf_via_elsevier_api(doi, path)
                source = "Elsevier API" if success else None
            for url, candidate_source in dict.fromkeys(candidates):
                if success:
                    break
                if self.download_pdf_file(url, path):
                    success, source = True, candidate_source
            # Query broader resolvers lazily; most PMC articles should not incur
            # three additional API calls after a successful Europe PMC download.
            for resolver, candidate_source in (
                (lambda: self.get_pdf_url_from_openalex(pmid=raw, doi=doi), "OpenAlex"),
                (lambda: self.get_pdf_url_from_unpaywall(doi), "Unpaywall"),
                (lambda: self.get_pdf_url_from_semantic_scholar(raw), "Semantic Scholar"),
            ):
                if success:
                    break
                url = resolver()
                if url and self.download_pdf_file(url, path):
                    success, source = True, candidate_source
            return {**base, "pmcid": pmcid or "N/A", "doi": doi or "N/A",
                    "pdf_status": "Downloaded (Open Access)" if success else "Unavailable/Paywall",
                    "pdf_path": str(path) if success else f"https://pubmed.ncbi.nlm.nih.gov/{raw}/",
                    "pdf_filename": filename if success else "N/A",
                    "download_source": source or "N/A"}

        results = []
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            futures = {executor.submit(process, item): item[0] for item in frame.iterrows()}
            for completed, future in enumerate(as_completed(futures), 1):
                try:
                    results.append(future.result())
                except Exception as error:
                    index = futures[future]
                    results.append({"source_row": int(index) + 1,
                                    "title": str(frame.loc[index, title_col]),
                                    "pubmed_id": str(frame.loc[index, pmid_col]),
                                    "pdf_status": "Error", "error": f"{type(error).__name__}: {error}"})
                if completed % 50 == 0:
                    print(f"Literature download progress: {completed}/{len(frame)}")
        summary = pd.DataFrame(results).sort_values("source_row")
        for index, row in summary.iterrows():
            pmid = str(row.get("pubmed_id", "")).removesuffix(".0")
            cached = pdf_dir / f"PMID{pmid}.pdf"
            if pmid.isdigit() and cached.exists() and cached.stat().st_size > 1024:
                with cached.open("rb") as handle:
                    valid = handle.read(5).startswith(b"%PDF")
                if valid and row.get("pdf_status") not in {
                    "Downloaded (Open Access)", "Cached (Local File)"
                }:
                    summary.loc[index, ["pdf_status", "pdf_path", "pdf_filename",
                                        "download_source"]] = [
                        "Cached (Recovered after concurrent completion)", str(cached),
                        cached.name, "cache_reconciled",
                    ]
        manifest = evidence_dir / f"{gene_symbol.lower()}_literature_evidence.csv"
        summary.to_csv(manifest, index=False)
        audit = {
            "source_csv": str(csv_path),
            "source_csv_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
            "n_rows": len(summary),
            "n_unique_pmids": int(summary.loc[summary["pubmed_id"] != "N/A", "pubmed_id"].nunique()),
            "status_counts": {str(key): int(value) for key, value in
                              summary["pdf_status"].value_counts(dropna=False).items()},
            "download_source_counts": {str(key): int(value) for key, value in
                                       summary["download_source"].fillna("N/A").value_counts().items()},
        }
        manifest.with_suffix(".manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        return manifest
