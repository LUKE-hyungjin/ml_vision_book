---
title: "카메라"
weight: 1
---

# 카메라

## 개요

Vision 시스템의 핵심 구성 요소인 카메라에 대해 알아봅니다.

---

## 이미지 센서

### CCD vs CMOS

| 특성 | CCD | CMOS |
|------|-----|------|
| 노이즈 | 낮음 | 보통~낮음 |
| 속도 | 느림 | 빠름 |
| 전력 | 높음 | 낮음 |
| 비용 | 비쌈 | 저렴 |
| 현재 추세 | 감소 | 대세 |

### 센서 크기

```
┌─────────────────────┐
│    Full Frame       │
│    (36 x 24 mm)     │
│  ┌───────────────┐  │
│  │   APS-C       │  │
│  │ (23 x 15 mm)  │  │
│  │  ┌─────────┐  │  │
│  │  │ 1 inch  │  │  │
│  │  │(13x9mm) │  │  │
│  │  └─────────┘  │  │
│  └───────────────┘  │
└─────────────────────┘
```

센서가 클수록:
- 더 많은 빛 수집 (저조도 성능)
- 더 얕은 심도 (배경 흐림)
- 높은 비용

### 해상도 선택

```
필요 해상도 = (검사 영역 크기) / (최소 검출 크기) × 여유 계수

예: 100mm 영역에서 0.1mm 결함 검출
→ 100 / 0.1 × 2 = 2000 pixels
→ 최소 2MP 카메라 필요
```

---

## 렌즈

### 초점 거리 (Focal Length)

```python
# 필요한 초점 거리 계산
def calculate_focal_length(sensor_size, object_size, working_distance):
    """
    sensor_size: 센서 크기 (mm)
    object_size: 촬영 영역 크기 (mm)
    working_distance: 카메라-피사체 거리 (mm)
    """
    focal_length = (sensor_size * working_distance) / object_size
    return focal_length

# 예: 1/2" 센서, 100mm 영역, 200mm 거리
fl = calculate_focal_length(6.4, 100, 200)  # ≈ 12.8mm
```

### F-Number (조리개)

```
밝기: 낮은 F = 밝음 (F1.4 > F2.8 > F5.6)
심도: 높은 F = 깊은 심도 (더 넓은 범위가 선명)
```

| F-Number | 밝기 | 심도 | 용도 |
|----------|------|------|------|
| F1.4-2.8 | 밝음 | 얕음 | 저조도, 고속 |
| F4-5.6 | 보통 | 보통 | 일반 |
| F8-16 | 어두움 | 깊음 | 검사, 정밀 측정 |

### 렌즈 종류

| 종류 | 특징 | 용도 |
|------|------|------|
| **일반 렌즈** | 원근감 있음, 저렴 | 일반 촬영 |
| **텔레센트릭** | 원근감 없음, 정확한 측정 | 정밀 검사, 측정 |
| **매크로** | 근접 촬영, 배율 1:1 이상 | 미세 결함 검사 |
| **광각** | 넓은 시야각 | 넓은 영역 감시 |

---

## 카메라 인터페이스

| 인터페이스 | 속도 | 거리 | 특징 |
|------------|------|------|------|
| **USB 3.0** | 5 Gbps | 5m | 간편, 저비용 |
| **GigE** | 1 Gbps | 100m | 긴 거리, 다중 연결 |
| **10GigE** | 10 Gbps | 100m | 고속, 고해상도 |
| **Camera Link** | 6.8 Gbps | 10m | 고속, 산업용 |
| **CoaXPress** | 12.5 Gbps | 40m | 초고속, 산업용 |

### GigE Vision

```python
# aravis 라이브러리 사용 예
import gi
gi.require_version('Aravis', '0.8')
from gi.repository import Aravis

# 카메라 연결
camera = Aravis.Camera.new(None)  # 첫 번째 발견된 카메라

# 설정
camera.set_region(0, 0, 1920, 1080)
camera.set_frame_rate(30)
camera.set_exposure_time(10000)  # μs

# 스트림 시작
stream = camera.create_stream()
camera.start_acquisition()

# 이미지 획득
buffer = stream.pop_buffer()
if buffer:
    data = buffer.get_data()
    stream.push_buffer(buffer)

camera.stop_acquisition()
```

---

## 글로벌 vs 롤링 셔터

### 글로벌 셔터

모든 픽셀 동시 노출:

```
시간 →
Row 1: [====노출====]
Row 2: [====노출====]
Row 3: [====노출====]
       ↑           ↑
      시작        종료
```

- 장점: 움직이는 물체 왜곡 없음
- 용도: 고속 이동 물체, 산업 검사

### 롤링 셔터

행별로 순차 노출:

```
시간 →
Row 1: [====노출====]
Row 2:   [====노출====]
Row 3:     [====노출====]
```

- 단점: 움직이는 물체 왜곡 (젤리 효과)
- 장점: 저비용, 고해상도

---

## 산업용 카메라 브랜드

| 브랜드 | 특징 |
|--------|------|
| **Basler** | 폭넓은 라인업, 합리적 가격 |
| **FLIR** | 열화상, 과학용 |
| **Allied Vision** | 고성능, 다양한 인터페이스 |
| **IDS** | USB3 Vision, 소프트웨어 우수 |
| **JAI** | 다중 센서, 라인 스캔 |
| **Teledyne DALSA** | 라인 스캔 전문 |

---

## 라인 스캔 카메라

### 원리

```
┌─────────────────────────────┐
│ ← 물체 이동 방향            │
│                             │
│ ■■■■■■■■■■ ← 1D 센서       │
│                             │
└─────────────────────────────┘
```

- 1D 센서가 이동하는 물체를 스캔
- 연속 공정에서 무한 길이 이미지 획득

### 용도

- 웹 검사 (필름, 직물, 종이)
- 컨베이어 벨트 검사
- 인쇄물 검사

### 설정 계산

```python
def calculate_line_rate(conveyor_speed_m_s, pixel_size_um, magnification):
    """
    라인 스캔 속도 계산

    conveyor_speed_m_s: 컨베이어 속도 (m/s)
    pixel_size_um: 픽셀 크기 (μm)
    magnification: 광학 배율
    """
    object_pixel_size_m = (pixel_size_um / magnification) * 1e-6
    line_rate = conveyor_speed_m_s / object_pixel_size_m
    return line_rate  # lines per second

# 예: 1 m/s, 5μm 픽셀, 0.5 배율
rate = calculate_line_rate(1.0, 5, 0.5)  # = 100,000 lines/s
```

---

## 3D 카메라

| 기술 | 원리 | 특징 |
|------|------|------|
| **스테레오** | 두 카메라 시차 | 저비용, 텍스처 필요 |
| **구조광** | 패턴 투사 | 정확, 실내용 |
| **ToF** | 빛 비행시간 | 빠름, 중거리 |
| **LiDAR** | 레이저 스캔 | 장거리, 정확 |

### Intel RealSense (스테레오 + 구조광)

```python
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

# Point Cloud 생성
pc = rs.pointcloud()
points = pc.calculate(depth_frame)
```

---

## 관련 콘텐츠

- [조명](/ko/docs/engineering/hardware/lighting) - 조명 설계
- [엣지 디바이스](/ko/docs/engineering/hardware/edge) - 처리 장치
- [기하학](/ko/docs/math/geometry) - 카메라 모델

