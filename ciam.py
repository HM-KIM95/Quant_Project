import os
import requests
from dotenv import load_dotenv

# 1) env 파일 로드
load_dotenv("DART.env")

API_KEY = os.getenv("DART_API_KEY")


def get_business_reports(corp_name: str):
    """DART OpenAPI를 사용하여 사업보고서(A001)를 조회하는 함수"""

    # 1) 기업 고유번호(corp_code) 조회
    url_corp = "https://opendart.fss.or.kr/api/corpCode.xml"

    # corpCode.xml은 압축파일로 제공됨 → 로컬 저장 후 파싱 필요
    # 그러나 더 쉬운 방법: 기업명으로 공시 검색(list API) 활용

    url_list = "https://opendart.fss.or.kr/api/list.json"

    params = {
        "crtfc_key": API_KEY,
        "corp_name": corp_name,
        "bgn_de": "19990101",
        "end_de": "20251231",
        "pblntf_detail_ty": "A001"  # 사업보고서만 조회
    }

    res = requests.get(url_list, params=params).json()

    return res


if __name__ == "__main__":
    print("사업보고서 조회 결과:")
    result = get_business_reports("삼성전자")
    print(result)