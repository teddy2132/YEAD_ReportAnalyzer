# ──────────────────────────────────────────────
#  extract_proposal.py
#    PDF 1건 → 구조화 JSON
#    필요 패키지: openai>=1.14.2, pymupdf, python-dotenv, tiktoken
# ──────────────────────────────────────────────
import os, json, re, fitz, openai, argparse
from dotenv import load_dotenv

# 0. 환경 변수
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. PDF → 텍스트
def pdf_to_text(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for pg in doc:
        txt = pg.get_text("text")
        if txt.strip():
            pages.append(txt)
        else:                                   # OCR fallback
            try:
                import pytesseract
                img = pg.get_pixmap(dpi=300).pil_tobytes("png")
                ocr = pytesseract.image_to_string(img, lang="kor+eng")
                pages.append(ocr)
            except Exception:
                pages.append("")
    return "\n".join(pages)

# 2. GPT 시스템 프롬프트 (스키마)
SYSTEM_PROMPT = """
반드시 아래 JSON 스키마 그대로만 채워서 반환하세요.
누락 시 null, 배열 필드는 배열로, 숫자는 단위 제거 후 숫자만:

{
"title":"","agency":"","year":"","company_name":"",
"problem_text":"","problem_metrics":"",
"solution_text":"","roadmap_present":"","roadmap_outline":"",
"patent_cnt":"","patent_details":"",
"paper_cnt":"","paper_refs":"",
"prototype_present":"","test_results":"",
"similar_case_cnt":"","similar_case_summary":"",
"market_size":"","market_growth":"","competitive_edge":"",
"expected_revenue":"","breakeven_year":"","roi_pct":"",
"core_members":"","prior_projects":"","infrastructure_summary":"",
"regulatory_ready":"","external_investment":"",
"risk_mitigation_present":"","policy_alignment":"",
"readability_score":"","diagram_cnt":"","summary_len_chars":"",
"tech_market_balance":"","logical_consistency":""
}
"""

# 3. JSON 안전 파서
def safe_json_loads(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"```json\s*|```", "", s, flags=re.I).strip()
    if "{" in s and "}" in s:
        s = s[s.find("{"): s.rfind("}") + 1]
    return json.loads(s)

# 4. GPT 호출 (동기)
def gpt_extract(text: str, model="gpt-4o-mini") -> dict:
    text = text[:12000]  # 토큰 12k 이내로 슬라이스
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text}
        ],
        temperature=0,
        max_tokens=3072
    )
    content = resp.choices[0].message.content
    if not content.lstrip().startswith("{"):
        print("◆ GPT 응답이 JSON 아님, 원문 일부 ↓")
        print(content[:800], "...\n")
    return safe_json_loads(content)

# 5. 추가 지표 계산
def postprocess(raw: str, js: dict) -> dict:
    sent = max(raw.count("."), 1)
    js["readability_score"] = round(len(raw.split()) / sent, 2)
    js["diagram_cnt"] = raw.lower().count("figure") + raw.lower().count("표")
    js["summary_len_chars"] = len(raw[:1000])
    tech = sum(raw.count(w) for w in ["특허","시제품","TRL","prototype","성능"])
    biz  = sum(raw.count(w) for w in ["시장","매출","고객","ROI","판로"])
    js["tech_market_balance"] = round(tech / (biz + 1), 2)
    js["logical_consistency"] = None
    return js

# 6. 파이프라인
def extract(pdf_path: str, out_json: str | None = None):
    print("▶ PDF 읽는 중… ", pdf_path)
    raw = pdf_to_text(pdf_path)

    print("▶ GPT 요약·추출 중…")
    data = gpt_extract(raw)
    data = postprocess(raw, data)

    print(json.dumps(data, ensure_ascii=False, indent=2))
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# 7. CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_path")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    extract(args.pdf_path, args.out)
