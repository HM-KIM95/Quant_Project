import os
from dotenv import load_dotenv
import dart_fss as dart

# 1) .env 파일 로드
load_dotenv("DART.env")

# 2) 환경 변수에서 API KEY 읽기
DART_API_KEY = os.getenv("DART_API_KEY")

if not DART_API_KEY:
    raise ValueError("❌ DART_API_KEY가 환경변수에서 로드되지 않았습니다.")

# 3) dart_fss에 API KEY 설정
dart.set_api_key(api_key=DART_API_KEY)

# 4) 테스트용 함수 예시
def get_corp_report(corp_name: str):
    """기업명으로 사업보고서 리스트 불러오기"""
    try:
        corp = dart.get_corp(corp_name)
        filings = corp.get_filings("사업보고서")
        return filings
    
    except Exception as e:
        return f"에러 발생: {e}"
    
# 실행 예시
if __name__ == "__main__":
    print("삼성전자 사업보고서 조회 결과:")
    result = get_corp_report("삼성전자")
    print(result)