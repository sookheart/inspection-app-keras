"""가죽 이상 탐지 모델을 사용하는 Streamlit 웹 앱."""

import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ── 페이지 설정 ──────────────────────────────────────────────────
# 이모지나 외부 이미지 파일 없이 사용할 수 있는 간단한 갈색 아이콘이다.
PAGE_ICON = Image.new("RGB", (32, 32), color=(105, 73, 48))

st.set_page_config(
    page_title="가죽 이상 탐지",
    page_icon=PAGE_ICON,
    layout="centered",
)

st.title("가죽 이상 탐지")
st.caption("가죽 이미지를 입력하면 AI 모델이 정상 여부와 예측 확률을 표시합니다.")


# ── 모델 및 추론 설정 ────────────────────────────────────────────
MODEL_PATH = "./weights/leather_model.keras"
INPUT_IMG_SIZE = (224, 224)
CLASSES = ["정상", "불량"]

FONT_PATH = "fonts/NanumGothic-Regular.ttf"
font_prop = fm.FontProperties(fname=FONT_PATH)

plt.title("검사 결과", fontproperties=font_prop)
plt.xlabel("분류", fontproperties=font_prop)
plt.ylabel("확률", fontproperties=font_prop)


# ─────────────────────────────────────────────────────────────────
# 1. 모델 로드
#    앱이 입력 변경 등으로 다시 실행되어도 캐시에 저장된 모델을 재사용한다.
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────
# 2. 이미지 전처리
#    기존 코드와 동일하게 RGB 변환, 224×224 크기 조정,
#    VGG16 전처리, 배치 차원 추가를 수행한다.
# ─────────────────────────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────────
# 3. 추론
#    기존 코드와 동일하게 sigmoid 출력값이 0.5보다 크면 불량으로 판정한다.
# ─────────────────────────────────────────────────────────────────
def predict(model, pil_img):
    arr = preprocess(pil_img)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob


# ─────────────────────────────────────────────────────────────────
# 4. 이미지 입력
# ─────────────────────────────────────────────────────────────────
input_mode = st.radio(
    "이미지 입력 방식",
    ["파일 업로드", "카메라 촬영"],
    horizontal=True,
)

if input_mode == "파일 업로드":
    image_file = st.file_uploader(
        "가죽 이미지 선택",
        type=["jpg", "jpeg", "png"],
    )
else:
    image_file = st.camera_input("가죽 이미지를 촬영하세요")

pil_img = None
if image_file is not None:
    try:
        pil_img = Image.open(image_file).convert("RGB")
        st.image(pil_img, caption="입력 이미지")
    except Exception as error:
        st.error(f"이미지를 읽을 수 없습니다: {error}")


# ─────────────────────────────────────────────────────────────────
# 5. 검사 실행 및 결과 표시
# ─────────────────────────────────────────────────────────────────
if st.button("검사 시작", type="primary", disabled=pil_img is None):
    try:
        with st.spinner("가죽 이미지를 검사하고 있습니다."):
            model = load_model()
            label, defect_prob = predict(model, pil_img)

        normal_prob = 1 - defect_prob

        if label == "정상":
            st.success("검사 결과: 정상입니다.")
        else:
            st.error("검사 결과: 불량입니다.")

        normal_column, defect_column = st.columns(2)
        normal_column.metric("정상 확률", f"{normal_prob:.1%}")
        defect_column.metric("불량 확률", f"{defect_prob:.1%}")

        chart_data = pd.DataFrame(
            {"확률": [defect_prob]},
            index=["불량"],
        )
        st.subheader("불량 확률")
        st.bar_chart(chart_data, y="확률", height=220)

    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"검사 중 오류가 발생했습니다: {error}")

