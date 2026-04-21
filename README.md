# Custom Semantic Gaussian Rasterizer (diff_gaussian_rasterization_solo)

이 레포지토리는 원본 3D Gaussian Splatting 프로젝트에서 분기된 다양한 rasterizer들을 융합하여 ObjectLoopSplat을 지원하기 위해 커스텀으로 구축된(JIT 동적 컴파일 전용) 래스터화 패키지입니다. 

## 📌 주요 출처 및 통합된 레포지토리들
이 시스템은 주로 다음 3가지 패키지의 기능들을 통합 및 개량하여 구성되었습니다.
1. **Original `gaussian_rasterizer` (3DGS Base)**: 기본 가우시안 렌더링, 전방향(forward) 및 후방향(backward) 패스 등 가장 안정적이고 핵심적인 3DGS 래스터화 기반 구조.
2. **`diff_gaussian_rasterization_w_pose`**: 카메라 포즈 최적화 및 SLAM을 위해 확장된 버전. 현재 ObjectLoopSplat은 PyTorch에서 직접 포즈를 최적화하므로 실시간 포즈 미분 로직(`dL_dtau`)은 제거되었으나, **`projmatrix_raw` 처리 및 깊이/투명도(depth/alpha) 렌더링의 구조적 기반**으로 활용되었습니다.
3. **`SemGaussSLAM/diff-gaussian-rasterization-w-depth_semantic`**: 각 점(Point)마다 고유한 Semantic 임베딩(`sh_sems`)과 Depth를 렌더링하도록 뷰 공간을 계산하는 래스터라이저.

---

## 🛠️ 주요 변경 사항 및 개선점 (Modifications)

### 1. `dL_dtau` 파라미터 및 포즈 미분 그래디언트의 완전한 제거
기존 `w_pose` 패키지에서는 카메라의 회전 및 평행이동에 직접적으로 연동되는 그래디언트인 `d_tau` (카메라 변환의 접선 벡터 공간)를 내부적으로 계산하고 반환했습니다.
그러나 현재 **ObjectLoopSplat 트래커(Tracker) 구조**에서는 파이토치의 `Tensor` 연산을 통해 카메라의 extrinsic matrix(c2w)를 곧바로 분리, 최적화하므로 해당 C++ 코드 상의 그래디언트 변환이 더 이상 필요하지 않습니다.
- **최적화**: C++ Extension(`ext.cpp`), Python Binder (`__init__.py`), `backward.cu`, `rasterizer.h` 등에 존재하던 모든 `dL_dtau` 계산을 소거하여 메모리 오버헤드와 연산 시간을 단축시켰습니다.

### 2. 시맨틱(Semantic) 피처의 렌더링 추가
- `backward.cu` 및 `forward.cu` 내에 `sh_sems`를 함께 래스터화하여 투영하는 기능을 추가했습니다. 이는 각 가우시안이 객체의 어느 시맨틱 카테고리에 해당하는지를 깊이 버퍼(depth buffer)뿐만 아니라 동시에 최적화할 수 있도록 돕습니다.
- `d_alpha` (투명도 조절 그래디언트): `SemGaussSLAM` 코드 베이스에서 확장된 기능으로, 시맨틱 레이블과 깊이맵을 기반으로 마스킹될 때 각 Gaussian이 투명도 값에 미치는 영향을 역전파할 수 있도록 추가되었습니다.

### 3. JIT (Just-In-Time) 동적 로딩 지원
시스템에 독립적으로 해당 라이브러리를 동적 로드하도록 `uv`와 `.venv` 환경에 맞춰 동작하도록 구성했습니다. 즉, 패키지를 정적으로 `pip install -e .` 하지 않아도, Python 레벨(예: `rasterizer_loader.py`)에서 런타임에 빌드되어 `import` 됩니다.

---

## 코드 내부의 핵심 변경 지점
- `cuda_rasterizer/backward.cu`: 가우시안 렌더링의 역전파 커널. (사용되지 않은 `dL_dtau` 역전파 소거 완료, `d_alpha` 및 깊이 최적화 지원 포함됨)
- `diff_gaussian_rasterization/__init__.py`: GPU에서 계산된 출력물들을 파이토치 Tensor로 매핑해주는 바인더 계층.