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

# 4) 기업명으로 사업보고서 조회
def get_corp_report(corp_name: str):
    """기업명으로 사업보고서 리스트 불러오기"""
    try:
        search_result = dart.search_corp(corp_name)
        
        if len(search_result) == 0:
            return f"기업 '{corp_name}'을 찾을 수 없습니다."
        
        corp = search_result[0] # 첫 번째 결과 선택
        
        # 사업보고서(annual) 검색
        
        filings = corp.get_filings(report_type="annual")
        
        return filings
    
    except Exception as e:
        return f"에러 발생: {e}"
    
# 실행 예시
if __name__ == "__main__":
    print("삼성전자 사업보고서 조회 결과:")
    result = get_corp_report("삼성전자")
    print(result)