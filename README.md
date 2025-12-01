
# SSAFY AI 챌린지 – 시각 QA(Visual MCQ) 모델 고도화 리포트

> 이미지 + 4지선다 선택지를 입력으로 받아, 정답(a/b/c/d)을 예측하는 VQA(Multiple-Choice) 모델을  
> **ChatGPT 기반 프롬프트 엔지니어링 + 하이퍼파라미터 튜닝**으로 고도화한 과정과 결과를 정리한 README입니다.

---

## 1. 프로젝트 개요

### 1.1 과제 정의

- **Task**  
  - 이미지 + 질문 + 보기 4개(a, b, c, d)를 입력으로 받아  
    **정답 선택지의 알파벳 1글자(a/b/c/d)** 를 출력하는 모델 학습  
- **목표**
  - 주어진 train 데이터로 지도학습을 수행하고  
    **리더보드 점수(accuracy)를 최대화**하는 것이 최종 목표  
- **최종 성능**
  - 개인 기록: `0.24176 → 0.86625`  
  - 팀 기록: `0.87860 → 0.94135`

### 1.2 개발 환경 & 일정

- **기간**: 2025.10.23 ~ 2025.10.27  
- **팀 구성**: 4인 (한상민, 김민수, 황가연, 양새하)  
- **환경**
  - Colab / SSAFY SSH GPU 서버  
  - Qwen2.5-VL 계열 Vision-Language 모델 + LoRA 미세조정

---

## 2. 베이스라인 설계

### 2.1 기본 파이프라인

1. 사전 학습 모델 & Processor 로드  
2. Custom Dataset / DataCollator로 Chat 형식 인코딩  
3. LoRA 기반 미세조정  
4. Inference 시 ChatTemplate + `generate`로 정답 한 글자 생성  

---

## 2.2 핵심 베이스라인 코드

### (1) 모델 & Processor 로드

```python
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_SIZE = 448
MAX_NEW_TOKENS = 8
SEED = 42

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=IMAGE_SIZE*IMAGE_SIZE,
    max_pixels=IMAGE_SIZE*IMAGE_SIZE,
    trust_remote_code=True,
)

base_model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
```

---

### (2) 프롬프트 템플릿 & Dataset

```python
SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)

def build_mc_prompt(question, a, b, c, d):
    return (
        f"{question}\n"
        f"(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )

class VQAMCDataset(Dataset):
    def __init__(self, df, processor, train=True):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.train = train

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        q = str(row["question"])
        a, b, c, d = str(row["a"]), str(row["b"]), str(row["c"]), str(row["d"])
        user_text = build_mc_prompt(q, a, b, c, d)

        messages = [
            {"role":"system","content":[{"type":"text","text":SYSTEM_INSTRUCT}]},
            {"role":"user","content":[
                {"type":"image","image":img},
                {"type":"text","text":user_text}
            ]}
        ]

        if self.train:
            gold = str(row["answer"]).strip().lower()
            messages.append({"role":"assistant",
                             "content":[{"type":"text","text":gold}]})

        return {"messages": messages, "image": img}
```

---

### (3) 기본 학습 루프

```python
GRAD_ACCUM = 4
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
num_training_steps = 5 * math.ceil(len(train_loader)/GRAD_ACCUM)
scheduler = get_linear_schedule_with_warmup(
    optimizer, int(num_training_steps*0.03), num_training_steps
)
scaler = torch.cuda.amp.GradScaler(enabled=True)

for epoch in range(5):
    for step, batch in enumerate(train_loader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        if step % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
```

---

## 3. 기능별 문제 상황 & 해결 전략  
아래부터 기능별로 **문제 → 해결 → 결과** 구조로 기술합니다.

---

## 3-1. 출력 형식 불안정 & 프롬프트 품질 개선

### 문제 상황
- 모델이 한 글자(a/b/c/d)만 출력해야 하지만  
  **문장, 공백, 개행, 마침표** 등이 함께 출력되는 오류 발생  
- 후처리에서 정답 추출 실패 가능성 존재  

### 해결 과정

#### (1) System / User 프롬프트 강화

```python
# AFTER
SYSTEM_INSTRUCT = (
    "You are a visual QA assistant for multiple-choice.\n"
    "- Output exactly ONE lowercase letter: a, b, c, or d\n"
    "- No punctuation/space/newline/explanation\n"
    "- Respond after the word 'Answer:' with a single letter"
)

def build_mc_prompt(question, a, b, c, d):
    return (
        f"{question}\n"
        f"(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "Answer: "
    )
```

#### (2) 디코딩 제약: a/b/c/d만 생성하도록 강제

```python
class OnlyABCDProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.allowed = set(tokenizer.convert_tokens_to_ids(list("abcd")))
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        idx = torch.tensor(list(self.allowed), device=scores.device)
        mask[:, idx] = scores[:, idx]
        return mask
```

### 결과
- 출력 포맷 오류 감소  
- 정답 추출 정확도 상승  
- 리더보드 accuracy 향상  

---

## 3-2. 라벨 설계 문제 – 프롬프트 전체를 따라치는 현상

### 문제 상황
- 초기 구조에서는 **system/user/assistant 모든 텍스트가 loss 대상**  
- 모델이 **정답 1글자**보다 프롬프트 전체를 학습함  

### 해결: 라벨 마스킹

```python
for i in range(enc["labels"].size(0)):
    cutoff = (enc_mask["input_ids"][i] != pad_id).sum()
    enc["labels"][i, :cutoff] = -100
```

### 결과
- 중요한 정답 토큰에 집중하도록 개선  
- 개인 accuracy **0.24 → 0.86** 점프  

---

## 3-3. 데이터 분할 문제 & 일반화 개선

### 문제
- 단순 90:10 split → 정답 분포가 고르지 않을 위험  
- valid 점수가 실제 일반화 성능 반영 X  

### 해결: Stratified Split 적용

```python
train_subset, valid_subset = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["answer"]
)
```

### 결과
- validation 안정화  
- 과적합 판단 쉬워짐  

---

## 3-4. 학습 안정성 강화  
### 문제
- 고용량 모델 + LoRA에서 loss 진동·폭발 발생  

### 해결

(1) Cosine Scheduler  
(2) Label Smoothing (ε=0.05)  
(3) EMA 적용  
(4) 학습률/웜업비율 미세조정  

### 결과
- loss 곡선 안정화  
- 리더보드 최고점 근접 (0.94대)  

---

## 3-5. 데이터 증강 – 보기(선지) 셔플

### 문제
- 항상 (a,b,c,d) 순서 → 위치 bias 발생  

### 해결

```python
random.shuffle(choices)
new_idx = label_map[label_orig]
new_label = "abcd"[new_idx]
```

### 결과
- 위치 bias 감소  
- 성능 안정성 증가  

---

## 3-6. 모델 업그레이드 (3B → 7B)

### 문제
- 3B 모델의 표현력이 상위권 도달에 부족  

### 해결

- SSH 서버에서 가능한 최대 용량 7B 로 업그레이드  

### 결과
- 전반적 accuracy 증가  
- LoRA+EMA+Smoothing 조합으로 팀 score 0.94 기록  

---

## 4. 최종 성과 & 회고

### 4.1 정량 성과
- 개인: **0.24176 → 0.86625**
- 팀: **0.87860 → 0.94135**

### 4.2 기술적 인사이트
- 라벨 마스킹이 성능의 절반 이상 결정  
- 프롬프트 + 디코딩 제약이 VLM 성능 극적으로 상승  
- 모델 용량 결정은 환경 제약(SSH GPU) 고려 필수  

### 4.3 개인 회고
- 팀 속도와 맞추는 데 어려움 → 소통을 통해 해결  
- 이번 프로젝트로 모델링 구조·하이퍼파라미터의 의미를 깊게 이해  
- 다음 목표: 전체 파이프라인을 스스로 설계 가능한 수준  

---

## 5. 재현 방법

1. 필수 패키지 설치 (transformers, accelerate, peft 등)
2. 데이터 로드 (train.csv, test.csv, 이미지)
3. Processor + 모델 불러오기
4. Dataset / DataCollator 구성 (라벨 마스킹 포함)
5. 학습 (Cosine + EMA + Smoothing)
6. 추론 (OnlyABCDProcessor 사용)
