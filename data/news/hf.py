import asyncio
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

#─────────────────────────────────────────────────────────
# (기존) URL → HTML 수집 함수들 ― 수정 없음
#─────────────────────────────────────────────────────────
async def _fetch_one(session: aiohttp.ClientSession,
                     url: str, idx: int,
                     title: str, symbol: str,
                     sem: asyncio.Semaphore,
                     timeout: int) -> dict | None:
    async with sem:
        try:
            async with session.get(url, timeout=timeout) as r:
                if r.status == 200:
                    html = await r.text()
                    return {
                        "ID":     idx,
                        "Symbol": symbol,
                        "Title":  title,
                        "url":    url,
                        "HTML":   html
                    }
        except Exception:
            return None

async def build_html_df_async(df: pd.DataFrame,
                              max_conc: int = 200,
                              timeout: int = 10,
                              user_agent: str = "Mozilla/5.0") -> pd.DataFrame:

    sem      = asyncio.Semaphore(max_conc)
    conn     = aiohttp.TCPConnector(limit_per_host=max_conc)
    headers  = {"User-Agent": user_agent}

    async with aiohttp.ClientSession(connector=conn, headers=headers) as session:
        tasks = [
            _fetch_one(session,
                       url=row["url"],
                       idx=int(idx),
                       title=row["headline"],
                       symbol=row["stock"],
                       sem=sem,
                       timeout=timeout)
            for idx, row in df.iterrows()
            if pd.notna(row["url"]) and row["url"] != ""
        ]

        rows = []
        for coro in tqdm_asyncio.as_completed(tasks, desc="Fetching"):
            r = await coro
            if r is not None:
                rows.append(r)

    result_df = pd.DataFrame(rows,
                             columns=["ID", "Symbol", "Title", "url", "HTML"])
    return result_df

#─────────────────────────────────────────────────────────
# main ― 청킹 + 주기적 저장
#─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, math, gc
    from pathlib import Path

    # ──────────────── 경로 및 파라미터 ────────────────
    infile      = sys.argv[1] if len(sys.argv) > 1 else (
        "hf://datasets/ashraq/financial-news/data/"
        "train-00000-of-00001-8ec327f23bbe0948.parquet"
    )
    out_prefix  = "./data/news/hf/hf_stock"                   # 덮어쓰기 방지용 prefix
    chunk_rows  = 10000     # 50,000 → 10,000  (필요하면 5,000·2,000까지 축소)
    max_conc    = 100       # 300     → 100     (동시 요청 수도 함께 완화)
    timeout     = 10                             # 요청 타임아웃

    # ──────────────── 입력 로드(메타만) ────────────────
    print(f"[INFO] loading '{infile}' …")
    df_all = pd.read_parquet(
        infile,
        columns=["headline", "url", "stock"]     # 메모리 최소화
    ).iloc[800000:850000]
    n_total = len(df_all)
    n_parts = math.ceil(n_total / chunk_rows)
    pad     = len(str(n_parts))                  # 1, 2, … → 01, 02 …

    print(f"[INFO] rows: {n_total:,}  |  chunks: {n_parts} "
          f"(≈{chunk_rows:,} rows/chunk)")

    # ──────────────── 청킹 루프 ────────────────
    for part_idx, start in enumerate(range(0, n_total, chunk_rows), start=1):
        end = min(start + chunk_rows, n_total)
        print(f"[INFO] chunk {part_idx}/{n_parts}  rows {start:,}–{end-1:,}")

        df_chunk = df_all.iloc[start:end]

        # 비동기 크롤링 실행
        try:
            df_html = asyncio.run(
                build_html_df_async(df_chunk, max_conc=max_conc, timeout=timeout)
            )
        except RuntimeError:                     # 이미 루프가 돌 경우
            loop = asyncio.get_event_loop()
            df_html = loop.run_until_complete(
                build_html_df_async(df_chunk, max_conc=max_conc, timeout=timeout)
            )

        # 파일명: hf_stock_0001.csv, hf_stock_0002.csv, …
        out_file = f"{out_prefix}_{part_idx + 113:0{pad}d}.parquet"
        df_html.to_parquet(out_file, index=False) 
        print(f"        saved → {out_file}  (rows: {len(df_html):,})")

        # 메모리 해제
        del df_chunk, df_html
        gc.collect()

    print(f"[DONE] all chunks processed; output files '{out_prefix}_*.csv'")
