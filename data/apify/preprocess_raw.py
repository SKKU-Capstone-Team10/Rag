import os
import json
import pandas as pd

def parse_and_save_json(json_path: str, csv_path: str):
    """
    단일 JSON 파일에서 지정된 키만 추출하여 CSV로 저장.
    
    Parameters:
    - json_path: 원본 JSON 파일 경로
    - csv_path: 출력할 CSV 파일 경로
    """
    # 1) JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    # 2) DataFrame 생성: 필요한 키만 뽑아서 dict 리스트로 변환
    filtered = []
    for rec in records:
        # 각 레코드마다 4개 키를 추출하되, 누락 시 빈 문자열로 처리
        filtered.append({
            'title':           rec.get('title', ''),
            'published':       rec.get('published', ''),
            'descriptionText': rec.get('descriptionText', ''),
            'storyPath':       rec.get('storyPath', '')
        })
    
    df = pd.DataFrame(filtered)
    
    # 3) 컬럼명 변경(Optional): 원한다면 here에서 rename
    # df = df.rename(columns={'storyPath': 'URL'})
    
    # 4) CSV로 저장
    df.to_csv(csv_path, index=False, encoding='utf-8')

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DIR  = os.path.join(BASE_DIR, "raw")
    CSV_DIR  = os.path.join(BASE_DIR, "../news")
    os.makedirs(CSV_DIR, exist_ok=True)

    # 5) raw 디렉터리 내 JSON 파일 순회
    for fname in os.listdir(RAW_DIR):
        if not fname.lower().endswith('.json'):
            continue

        json_file = os.path.join(RAW_DIR, fname)
        csv_file  = os.path.join(CSV_DIR, os.path.splitext(fname)[0] + '.csv')

        parse_and_save_json(json_file, csv_file)
        print(f"Saved: {csv_file}")
