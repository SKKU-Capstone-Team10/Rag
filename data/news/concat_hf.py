import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import re, gc, os
from tqdm import tqdm

# ─────────── 설정 ───────────
ROOT_DIR  = Path("./data/news/hf")
OUT_FILE  = ROOT_DIR / "hf_stock_all.parquet"
BATCH_SZ  = 16_000          # RAM 여유없으면 8_000·4_000으로
COMPR     = "snappy"

# ─────────── 대상 파일 목록 ───────────
pattern = re.compile(r"hf_stock_(\d+)\.parquet$")
all_files = sorted(f for f in ROOT_DIR.glob("hf_stock_*.parquet")
                   if pattern.match(f.name))

print(f"[INFO] 후보 파일: {len(all_files)}개")

# ───── (1) 사전 검증 – Parquet 파일만 추려내기 ─────
good_files, bad_files = [], []

for f in all_files:
    try:
        # 너무 작은 파일(헤더만 있거나 0바이트)도 제외
        if os.path.getsize(f) < 100:                       # 100B 미만이면 skip
            raise ValueError("file size < 100B")
        pq.ParquetFile(f)                                  # 메타 열어보기
        good_files.append(f)
    except Exception as e:
        bad_files.append(f)
        print(f"[WARN] {f.name} 건너뜀 – {e}")

print(f"[INFO] 정상 Parquet: {len(good_files)}개 / "
      f"손상·비호환: {len(bad_files)}개")

if not good_files:
    raise RuntimeError("병합할 유효 Parquet 파일이 없습니다!")

# ───── (2) 첫 파일 스키마로 ParquetWriter 준비 ─────
first_pf = pq.ParquetFile(good_files[0])
writer   = pq.ParquetWriter(
    OUT_FILE,
    schema=first_pf.schema_arrow,
    compression=COMPR
)

# ───── (3) 파일별·배치별 스트리밍 병합 ─────
for f in tqdm(good_files, desc="Merging Files"):
    try:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=BATCH_SZ):
            writer.write_table(pa.Table.from_batches([batch]))
    except Exception as e:
        print(f"[WARN] {f.name} 병합 중 오류 – 건너뜀: {e}")
    finally:
        del pf, batch
        gc.collect()

writer.close()
print(f"[DONE] 병합 완료 → {OUT_FILE}")
if bad_files:
    print(f"[NOTE] 손상·스킵된 파일 목록: {[p.name for p in bad_files]}")
