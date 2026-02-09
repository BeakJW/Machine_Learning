"""
PDF 암호 해제 도구
- 암호로 보호된 PDF를 해제해서 새 파일로 저장
- 여러 PDF 일괄 처리 지원
- 출력 파일명 자동 생성 (예: sample.pdf -> sample_unlock.pdf)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple
from PyPDF2 import PdfReader, PdfWriter


# =====================
# ✅ 여기만 주로 바꾸면 됨
# =====================
BASE_DIR = Path("/Users/jinwoo/Machine_Learning")

# 방식 A) 파일명 리스트만 바꾸기 (가장 단순)
PDF_NAMES: List[str] = [
    "03-BaggingㅣRandomForest.pdf",
]

# 방식 B) 패턴으로 자동 선택 (PDF_NAMES가 비어있으면 이걸 사용)
PATTERN = "03-BaggingㅣRandomForest.pdf"  # 예: "*풀이*.pdf"

PASSWORD = "it123!@#"

# 출력 규칙
OUT_DIR = BASE_DIR / "unlocked"   # 출력 폴더 (없으면 자동 생성)
SUFFIX = "_unlock"                # 파일명 뒤에 붙일 접미사
COPY_IF_NOT_ENCRYPTED = False     # 암호 없으면 복사본 만들지 여부
# =====================


def _safe_output_path(input_path: Path, out_dir: Path, suffix: str) -> Path:
    """
    중복 파일명이 있으면 (1), (2) 붙여서 안전하게 저장 경로 생성
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{input_path.stem}{suffix}.pdf"
    candidate = out_dir / base_name
    if not candidate.exists():
        return candidate

    i = 1
    while True:
        candidate = out_dir / f"{input_path.stem}{suffix} ({i}).pdf"
        if not candidate.exists():
            return candidate
        i += 1


def remove_pdf_password(input_pdf_path: Path, output_pdf_path: Path, password: str) -> Tuple[bool, str]:
    """
    returns: (성공여부, 메시지)
    """
    try:
        reader = PdfReader(str(input_pdf_path))

        if not reader.is_encrypted:
            if COPY_IF_NOT_ENCRYPTED:
                writer = PdfWriter()
                for page in reader.pages:
                    writer.add_page(page)
                with open(output_pdf_path, "wb") as f:
                    writer.write(f)
                return True, "암호 없음 → 복사본 저장"
            return False, "암호 없음 → 스킵"

        # PyPDF2 decrypt 결과는 버전에 따라 int/boolean 형태일 수 있음
        dec = reader.decrypt(password)
        if not dec:
            return False, "암호 불일치"

        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)

        with open(output_pdf_path, "wb") as f:
            writer.write(f)

        return True, "암호 해제 저장 완료"

    except FileNotFoundError:
        return False, "파일 없음"
    except Exception as e:
        return False, f"오류: {e}"


def collect_inputs(base_dir: Path, pdf_names: List[str], pattern: str) -> List[Path]:
    """
    - pdf_names가 있으면 그걸 우선 사용
    - 없으면 pattern(glob)로 수집
    """
    if pdf_names:
        paths = [base_dir / name for name in pdf_names]
    else:
        paths = sorted(base_dir.glob(pattern))

    # 실제로 존재하는 것만 남김
    return [p for p in paths if p.exists() and p.is_file() and p.suffix.lower() == ".pdf"]


def main() -> None:
    inputs = collect_inputs(BASE_DIR, PDF_NAMES, PATTERN)

    if not inputs:
        print("✗ 처리할 PDF가 없음")
        print(f"- BASE_DIR: {BASE_DIR}")
        print(f"- PDF_NAMES: {PDF_NAMES}")
        print(f"- PATTERN: {PATTERN}")
        return

    print(f"📄 대상 PDF 수: {len(inputs)}")
    print(f"📁 출력 폴더: {OUT_DIR}")

    ok_cnt = 0
    skip_cnt = 0
    fail_cnt = 0

    for in_path in inputs:
        out_path = _safe_output_path(in_path, OUT_DIR, SUFFIX)
        ok, msg = remove_pdf_password(in_path, out_path, PASSWORD)

        if ok:
            ok_cnt += 1
            print(f"✓ {in_path.name} -> {out_path.name} | {msg}")
        else:
            # 암호 없음 스킵도 실패처럼 보이기 싫으면 분리
            if "스킵" in msg:
                skip_cnt += 1
                print(f"↷ {in_path.name} | {msg}")
            else:
                fail_cnt += 1
                print(f"✗ {in_path.name} | {msg}")

    print("\n--- 결과 ---")
    print(f"✓ 성공: {ok_cnt}")
    print(f"↷ 스킵: {skip_cnt}")
    print(f"✗ 실패: {fail_cnt}")


if __name__ == "__main__":
    main()
