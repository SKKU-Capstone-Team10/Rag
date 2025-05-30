import csv
from pathlib import Path


def parse_note_to_csv(txt_path: str | Path, csv_path: str | Path) -> None:
    """
    메모장(txt) 파일을 읽어 각 ETF 블록에서 Symbol / Name을 추출한 뒤
    ID,Symbol,Name 열을 가진 CSV 파일로 저장한다.

    Parameters
    ----------
    txt_path : str | Path
        원본 메모장 파일 경로
    csv_path : str | Path
        생성할 CSV 파일 경로
    """
    txt_path = Path(txt_path)
    csv_path = Path(csv_path)

    # 1) 파일 읽기 ― \r\n, \n 모두 허용
    raw_text = txt_path.read_text(encoding="utf-8").replace("\r\n", "\n")

    # 2) 블록 단위(빈 줄이 두 번 연속)로 분리
    blocks = [blk.strip() for blk in raw_text.split("\n\n") if blk.strip()]

    records: list[tuple[int, str, str]] = []
    for idx, blk in enumerate(blocks, start=1):
        lines = blk.splitlines()

        # Symbol, Name만 필요. 다른 정보는 무시
        try:
            symbol = lines[0].strip()
            name   = lines[1].strip().replace("\"", '')
        except IndexError:
            raise ValueError(
                f"{idx}번째 블록에 두 줄 미만의 데이터가 있어 파싱할 수 없습니다."
            )

        records.append((idx, symbol, name))

    # 3) CSV로 저장
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Symbol", "Name"])   # 헤더
        writer.writerows(records)

    print(f"✔ {csv_path} 저장 완료 – {len(records)}개 레코드")

if __name__ == "__main__":
    # 실행 예시
    parse_note_to_csv(
        txt_path="./data/tradingview/stock_raw.txt",
        csv_path="./data/tradingview/stock.csv"
    )
    parse_note_to_csv(
        txt_path="./data/tradingview/etf_raw.txt",
        csv_path="./data/tradingview/etf.csv"
    )
