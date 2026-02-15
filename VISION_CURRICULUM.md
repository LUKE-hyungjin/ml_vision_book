# Vision Engineer Curriculum (판매형 학습 로드맵)

이 문서는 `STRUCTURE.md`를 기반으로, **지식이 없는 입문자도 따라올 수 있게** 학습 순서/완료 기준/보강 기준을 정리한 운영 문서입니다.

---

## 0) 목표

- 목표: Hugo 블로그 하나로 **Vision 엔지니어 실무 입문~중급**까지 도달하게 만든다.
- 원칙: 기존 문서를 지우지 않고, **보강(설명/예시/디버깅/FAQ)** 중심으로 완성도를 높인다.
- 언어: ko/en 대칭 유지 (경로, 링크, 핵심 섹션 구조).

---

## 1) 학습 레벨(처음부터)

## Level 0 — 입문 브리지 (수학/텐서 감각)
- 범위: `math/linear-algebra`, `math/calculus`, `math/probability`
- 완료 기준:
  - 벡터/행렬 shape를 읽고 연산 흐름을 설명할 수 있음
  - softmax, cross-entropy, gradient를 말로 설명 가능
  - 간단한 numpy/pytorch 텐서 코드에서 shape 오류를 디버깅 가능
- 문서 보강 포인트:
  - “왜 필요한가” + 생활 비유 + 최소 수식 + 기호 설명 + 작은 숫자 예시

## Level 1 — 딥러닝 부품 이해
- 범위: `components/convolution`, `components/normalization`, `components/activation`, `components/training/*`
- 완료 기준:
  - Conv/Norm/Activation의 역할을 파이프라인으로 설명 가능
  - optimizer/loss/regularization을 선택 이유와 함께 설명 가능
- 문서 보강 포인트:
  - 실패 패턴(학습 불안정, NaN, 과적합)과 점검 체크리스트

## Level 2 — Vision 핵심 태스크
- 범위: `components/detection`, `architecture/detection`, `architecture/segmentation`, `task/*`
- 완료 기준:
  - IoU/Anchor/NMS 연결 설명 가능
  - detection/segmentation 지표(AP, IoU 등) 해석 가능
- 문서 보강 포인트:
  - 임계값 변화에 따른 TP/FP 변화 표, 오동작 사례(과억제/과검출)

## Level 3 — Transformer/VLM/생성모델
- 범위: `components/attention`, `architecture/transformer`, `architecture/multimodal`, `architecture/generative`
- 완료 기준:
  - self/cross-attention, positional encoding, diffusion 핵심 직관 설명 가능
  - 실무 디버깅(마스크, mixed precision, cache, shape) 가능
- 문서 보강 포인트:
  - 수식-구현-디버깅의 1:1 매핑

## Level 4 — 실무 엔지니어링
- 범위: `engineering/data`, `engineering/deployment`, `engineering/hardware`
- 완료 기준:
  - 데이터 파이프라인/서빙/최적화/하드웨어 제약을 설명 가능
  - 프로젝트 배포 체크리스트 작성 가능
- 문서 보강 포인트:
  - 운영 체크리스트, 장애 대응 가이드, 튜닝 순서

---

## 2) 문서 보강 공통 템플릿 (초보자 친화)

각 문서는 가능하면 아래 순서를 따른다.

1. 왜 필요한가? (문제상황)
2. 한 줄 요약
3. 핵심 수식 + 기호 설명
4. 직관(비유/그림)
5. 최소 구현 코드
6. 실무 디버깅 체크리스트
7. 자주 하는 실수(FAQ)
8. 관련 콘텐츠(다음 학습 링크)

---

## 3) 피드백 루프 (학습→부족점→보강)

각 사이클에서 아래 4가지를 남긴다.

1. 오늘 공부한 개념(Study)
2. 부족한 점/오해 가능 지점(Gaps)
3. 보강한 문서와 보강 내용(Patch)
4. 다음 사이클 제안(Next)

> 핵심: "문서 수정"이 아니라, **학습자 이해 실패 지점을 줄이는 설계 개선**이 목적.

---

## 4) 판매형 품질 기준 (출시 전 체크)

- [ ] 지식 0 기준으로 읽어도 용어가 막히지 않는다.
- [ ] 모든 핵심 수식에 기호 설명이 있다.
- [ ] 문서마다 최소 1개 실행 가능한 코드 예제가 있다.
- [ ] 문서마다 디버깅 체크리스트가 있다.
- [ ] ko/en 경로와 구조가 대칭이다.
- [ ] 링크 규칙(입구/콘텐츠)이 깨지지 않는다.
- [ ] Hugo 빌드가 통과한다.

---

## 5) 운영 원칙

- 기존 작성물은 삭제보다 보강을 우선한다.
- 내용이 잘못된 경우에만 최소 수정으로 교정한다.
- 한 사이클에 1~2개 문서만 깊게 개선한다.
- `STRUCTURE.md`의 선수지식 체인을 항상 우선한다.
