---
title: "하드웨어"
weight: 3
bookCollapseSection: true
---

# 하드웨어

Vision 시스템 구축에 필요한 하드웨어 지식입니다.

## 주요 내용

| 주제 | 설명 |
|------|------|
| [카메라](/ko/docs/engineering/hardware/camera) | 센서, 렌즈, 인터페이스 |
| [조명](/ko/docs/engineering/hardware/lighting) | 조명 설계 기초 |
| [엣지 디바이스](/ko/docs/engineering/hardware/edge) | Jetson, 엣지 AI |

---

## Vision 시스템 구성

```
┌─────────────────────────────────────────────────────────┐
│                    Vision System                         │
│                                                          │
│   조명 ──→ 피사체 ──→ 렌즈 ──→ 센서 ──→ 프로세서       │
│                                                          │
│   Light    Object     Lens    Camera    Edge/Server     │
└─────────────────────────────────────────────────────────┘
```

---

## 하드웨어 선택 가이드

| 요소 | 고려사항 |
|------|----------|
| **해상도** | 검출할 결함/객체 크기 |
| **프레임레이트** | 이동 속도, 처리량 |
| **조명** | 반사, 그림자, 균일성 |
| **인터페이스** | USB, GigE, Camera Link |
| **비용** | 예산 대비 성능 |

