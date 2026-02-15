---
name: vision-study-loop
description: Study computer vision curriculum from scratch using STRUCTURE.md and create feedback artifacts (what was learned, gaps, next actions) that directly drive Hugo doc improvements. Use when running beginner-first curriculum cycles.
---

# Vision Study Loop

## 목적

초보자 기준으로 비전 엔지니어 커리큘럼을 학습하고,
학습 중 드러난 부족점을 Hugo 문서 보강으로 연결한다.

## 입력
- `STRUCTURE.md`
- `CLAUDE.md`
- `VISION_CURRICULUM.md`
- 해당 사이클의 대상 문서(ko/en)

## 출력
- 학습 요약(무엇을 공부했는가)
- 부족점 목록(무엇이 어렵고 왜 막히는가)
- 문서 보강 패치(어디를 어떻게 고쳤는가)
- 다음 사이클 1개 제안

## 실행 절차
1. 커리큘럼 레벨 선택 (Level 0~4)
2. 대상 주제 1개 선택 (필요시 ko/en 2파일)
3. 부족점 2~4개 추출
4. 보강 패치 적용 (비삭제 원칙)
5. `hugo --gc --minify` 검증
6. 보고

## 부족점 추출 체크리스트
- 용어가 설명 없이 등장하는가?
- 수식 기호 의미가 빠졌는가?
- 예제가 실행 가능한가?
- 디버깅 포인트가 있는가?
- ko/en 구조가 대칭인가?

## 금지
- 기존 내용을 대량 삭제하는 리라이트
- i18n 비대칭 생성
- 링크 규칙 위반
