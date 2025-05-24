import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

if __name__ == "__main__":
    PARQUET_FILE = "data/news/hf/hf_stock_all.parquet"
    BATCH_SIZE   = 10_000            # 한 배치에 10 000행씩만 읽어올 예

    # ParquetFile 객체로 메타데이터만 로드
    pf = pq.ParquetFile(PARQUET_FILE)

    # 필요한 칼럼만 지정하면 RAM 부담을 더 줄일 수 있습니다.
    columns = ["ID", "Symbol", "Title", "url"]

    # 배치 단위로 순차 처리
    for i, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE, columns=columns), start=1):
        df_chunk = pa.Table.from_batches([batch]).to_pandas()
        print(f"[Batch {i}] rows: {len(df_chunk)}  memory: {df_chunk.memory_usage(deep=True).sum()/1024**2:.1f} MiB")
        # ── 여기에 chunk 단위 처리 로직(필터링·집계·파일 저장 등)을 넣으세요 ──
        # 예: df_chunk.to_csv(f"chunk_{i:03d}.csv", index=False)

        # 메모리 해제
        del df_chunk
    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        df_chunk = pa.Table.from_batches([batch], schema=pf.schema_arrow).to_pandas()
        print(df_chunk.info())
        break