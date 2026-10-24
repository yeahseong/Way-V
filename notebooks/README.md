# 데이터셋 구축 · 학습 노트북

논문의 실험에 쓴 **학습 데이터 8만 장과 23개 모델 테스트셋을 직접 만들고, 학습을 실행한** 기록입니다.

데이터 구축과 초기 실험은 Google Colab에서 진행했고, 최종 학습은 연구실에서 빌린 RTX 3090 환경에서 돌렸습니다.

| 노트북 | 내용 | 결과물 |
|---|---|---|
| [00_cross_dataset_baseline](00_cross_dataset_baseline.ipynb) | **공개 데이터셋 초기 실험 + 교차 데이터셋 검증** (2026.03) | 방향 전환의 근거 |
| [01_collect_datasets](01_collect_datasets.ipynb) | FFHQ 실제 얼굴 · StyleGAN2/3 생성 얼굴 수집, 폴더 구조 설계 | 초기 학습 데이터 |
| [02_generate_progan_faces](02_generate_progan_faces.ipynb) | **ProGAN으로 가짜 얼굴 40,000장 생성** + FFHQ 실제 40,000장 수집 | 최종 학습셋 8만 장 |
| [03_build_diffusion_testset](03_build_diffusion_testset.ipynb) | SDXL · FLUX1_dev · FLUX1_schnell 테스트셋 구축 | Diffusion 테스트셋 12,000장 |
| [04_train_dffreq](04_train_dffreq.ipynb) | DFFreq 학습 실행 (Colab A100) | 학습된 탐지 모델 |
| [05_verify_dataset](05_verify_dataset.ipynb) | 클래스별 이미지 개수 검증 | 12개 모델 × (Real 2,000 / Fake 2,000) 확인 |

## 데이터 구성

```
학습  Real  FFHQ 40,000        Fake  ProGAN 40,000      (train : val = 9 : 1)
테스트 모델별 Real 2,000 / Fake 2,000  ·  GAN 12종 · Diffusion 4종 · Face Swap 5종 · Personalized 2종
```

## 작업하면서 해결한 것들

**공개 데이터셋의 한계를 수치로 확인** — So-Fake-Set 30,000장으로 100 epoch 학습했지만 가짜 탐지 정확도는 49.4%에 그쳤습니다. 학습에 쓰지 않은 FaceForensics++·DeepDetect로 교차 테스트해도 55.9%로 같은 문제가 재현됐고, 라벨 반전 검증까지 거쳐 설정 실수가 아님을 확인했습니다. 생성 방식이 뒤섞인 데이터로는 공통 아티팩트를 학습할 수 없다고 보고, 도메인과 생성 모델을 통제하는 쪽으로 방향을 바꿨습니다. ([00](00_cross_dataset_baseline.ipynb))

**원본 코드의 환경 의존성 제거** — DFFreq 공식 코드에 개발자 PC 경로(`/home/HDD/yjz/...`)가 하드코딩되어 있어 설정 파일을 직접 수정했고, Drive 동기화 중 잘린 체크포인트는 에포크를 역순으로 탐색해 정상 파일을 찾아 썼습니다. ([00](00_cross_dataset_baseline.ipynb))

**학습 데이터가 테스트에 섞이지 않게 분리** — 테스트용 Real 이미지도 FFHQ에서 가져와야 했는데, 학습에 쓴 것과 겹치면 성능이 부풀려집니다. HuggingFace 스트리밍에 `skip(40000).take(12000)`을 적용해 학습에 쓴 앞 4만 장을 건너뛰고 그다음 1만 2천 장만 테스트용으로 가져왔습니다. ([01](01_collect_datasets.ipynb))

**NVlabs 접근 차단 우회** — ProGAN 공식 저장소와 사전학습 모델 CDN이 모두 403을 반환해, Meta `pytorch_GAN_zoo`의 동일 계열 모델(celebAHQ-512)로 대체했습니다. PyTorch 버전 차이로 발생한 옵티마이저 로드 오류는 해당 클래스를 로드 동안만 감싸 해결했습니다. ([02](02_generate_progan_faces.ipynb))

**고주파 보존을 위한 PNG 통일** — 이 연구는 DWT·FFT로 고주파 아티팩트를 보는데, JPEG 재압축은 바로 그 고주파를 깎아냅니다. 모든 데이터를 무손실 PNG로 변환해 판단 근거가 훼손되지 않도록 했습니다. ([03](03_build_diffusion_testset.ipynb))

**Drive I/O 병목 제거** — 8만 장을 Drive에 마운트한 채 학습하면 데이터 로딩이 병목이 되어, 전체를 Colab 로컬 디스크로 복사한 뒤 학습했습니다. 재사용을 위해 `tar.gz`로 묶어 보관했습니다. ([04](04_train_dffreq.ipynb))

**평가 균형 맞추기** — 모델마다 Real 2,000 / Fake 2,000으로 1:1을 맞추고, 특정 프롬프트나 구도로 치우치지 않도록 무작위 샘플링했습니다. 학습 전에는 항상 개수 검증을 먼저 돌렸습니다. ([03](03_build_diffusion_testset.ipynb), [05](05_verify_dataset.ipynb))

## 공개하면서 정리한 것

- 원본 노트북에 **Kaggle API 키가 코드에 직접 들어 있었습니다.** 환경변수 주입 방식으로 바꾸고 기존 키는 폐기했습니다.
- 진행률 바와 대용량 출력은 핵심 로그만 남기고 정리했습니다.
- 경로는 작업 당시 Colab/Drive 기준이라 그대로 실행되지 않습니다. 재현하려면 `dataroot` 경로를 환경에 맞게 바꿔야 합니다.
