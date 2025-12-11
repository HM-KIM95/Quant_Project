import os
import zipfile
import xml.etree.ElementTree as ET
import requests
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv("DART.env")
API_KEY = os.getenv("DART_API_KEY")

# corpCode.xml 다운로드 경로
CORP_CODE_ZIP = "corpCode.zip"
CORP_CODE_XML = "CORPCODE.xml"


def download_corp_code():
    """OpenDART에서 corpCode.zip을 다운로드하여 XML 파일로 저장"""
    if os.path.exists(CORP_CODE_XML):
        return  # 이미 존재하면 다운로드 생략

    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={API_KEY}"
    response = requests.get(url)

    with open(CORP_CODE_ZIP, "wb") as f:
        f.write(response.content)

    # ZIP 압축 해제
    with zipfile.ZipFile(CORP_CODE_ZIP, 'r') as zip_ref:
        zip_ref.extractall(".")


def find_corp_code(corp_name: str):
    """기업명을 기반으로 corp_code를 조회"""
    tree = ET.parse(CORP_CODE_XML)
    root = tree.getroot()

    for corp in root.findall("list"):
        if corp.find("corp_name").text == corp_name:
            return corp.find("corp_code").text

    return None


def get_business_reports(corp_name: str):
    """corp_code 기반 사업보고서 조회"""
    download_corp_code()

    corp_code = find_corp_code(corp_name)

    if not corp_code:
        return f"기업 '{corp_name}'을 찾을 수 없습니다."

    url = "https://opendart.fss.or.kr/api/list.json"

    params = {
        "crtfc_key": API_KEY,
        "corp_code": corp_code,
        "bgn_de": "19990101",
        "end_de": "20251231",
        "pblntf_detail_ty": "A001"
    }

    res = requests.get(url, params=params).json()
    return res


if __name__ == "__main__":
    print("사업보고서 조회 결과:")
    print(get_business_reports("삼성전자"))