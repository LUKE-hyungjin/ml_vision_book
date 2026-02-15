---
name: hugo-improvement-planner
description: Run a curriculum-first, feedback-driven Hugo content improvement loop for ml_vision_book. Use when improving docs for beginner-to-vision-engineer learning outcomes: study a topic from STRUCTURE.md, identify learner gaps, and patch ko/en docs without deleting existing content.
---

# Hugo Improvement Planner (Curriculum + Feedback Loop)

이 스킬의 목적은 단순 문장 수정이 아니라,
**학습자 부족점(막힘)을 줄이는 보강**으로 Hugo 블로그 완성도를 높이는 것입니다.

## Core goal

- `STRUCTURE.md` 기반으로 "처음부터" 학습 흐름을 따른다.
- 기존 콘텐츠를 지우지 말고 보강한다.
- ko/en 대칭(i18n)을 유지한다.
- 한 사이클에 1~2개 문서를 깊게 개선한다.

## Read order
1. `STRUCTURE.md`
2. `CLAUDE.md`
3. `VISION_CURRICULUM.md`
4. 대상 문서(ko/en 쌍)

## Non-destructive rules (중요)
- 기존 섹션/문단은 원칙적으로 삭제하지 않는다.
- 잘못된 내용 교정이 필요한 경우만 최소 수정한다.
- 이미 있는 설명 위에 **초보자용 브릿지 설명/예시/체크리스트**를 보강한다.

## Topic selection rules
- `STRUCTURE.md`에서 `[ ]` 항목 또는 불균형(ko/en 비대칭) 우선.
- 선수지식 체인이 끊긴 구간을 우선.
- 동일 주제에 과집중하지 말고(예: attention만 반복), 커리큘럼 레벨을 순환한다.

## Per-cycle workflow

1. **Study**
   - 오늘 다룰 개념을 학습자 관점으로 재해석한다.
   - "초보자가 어디서 막히는지" 2~4개를 뽑는다.

2. **Gap extraction**
   - 아래 유형으로 부족점을 식별한다.
     - 용어 설명 없음
     - 수식 기호 설명 부족
     - 수식↔코드 연결 부족
     - 디버깅/실무 체크리스트 부재
     - ko/en 구조 불일치

3. **Patch (ko/en 동시 보강)**
   - 경로 대칭으로 문서를 보강한다.
   - 우선 보강 블록:
     - Why / Intuition / Symbol meanings
     - Minimal runnable code
     - Practical checklist / Failure patterns

4. **Validate**
   - `hugo --gc --minify` 실행
   - 실패 시 직전 변경을 되돌리고 원인 요약

5. **Report**
   - 아래 템플릿으로 짧게 보고한다.

## Writing constraints
- 링크는 절대경로 + 언어 prefix (`/ko/...`, `/en/...`).
- front matter 유효성 유지(`title`, `weight`, `math` 등).
- 입구↔콘텐츠 링크 규칙 위반 금지(`CLAUDE.md` 준수).

## Cycle report template

- 수정 파일
  - `...`
- 핵심 수정 내용
  - Study 관점에서 무엇을 보강했는지
  - Gap(부족점) 2~4개와 대응 보강
- 빌드 결과
  - 성공/실패
- 다음 사이클 제안
  - 커리큘럼 흐름상 다음 1개 주제

## Quality bar (판매형)
- 지식 0 학습자가 읽어도 용어 장벽이 낮아진다.
- 수식은 기호 설명과 함께 제공된다.
- 최소 1개 실행 가능한 예시/점검 포인트가 있다.
- 문서가 "설명"만이 아니라 "실전 디버깅"까지 연결된다.
