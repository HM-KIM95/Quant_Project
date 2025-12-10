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


def get_corp_report(corp_name: str):
    """기업명으로 사업보고서를 조회하는 함수"""
    try:
        # 기업 리스트 불러오기
        corp_list = dart.get_corp_list()
        corp = corp_list.find_by_corp_name(corp_name, exactly=False)

        if not corp:
            return f"기업 '{corp_name}'을 찾을 수 없습니다."

        target = corp[0]

        # 기업 고유 코드
        corp_code = target.corp_code

        # 사업보고서 조회 (pblntf_detail_ty = 사업보고서)
        filings = dart.api.filings.get_list(
            corp_code=corp_code,
            bgn_de='19990101',
            end_de='20251231',
            pblntf_detail_ty='A001'  # 사업보고서
        )

        return filings

    except Exception as e:
        return f"에러 발생: {e}"


# 테스트 실행
if __name__ == "__main__":
    print("사업보고서 조회 결과:")
    result = get_corp_report("삼성전자")
    print(result)