from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import shap
except Exception:
    shap = None


st.set_page_config(
    page_title="ARDS Risk Screening",
    page_icon="AI",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "evalresult_enet.pkl"

FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "ventilator_flag": {
        "type": "categorical",
        "label": "Mechanical Ventilation",
        "options": ["No", "Yes"],
        "default": "Yes",
    },
    "crrt_flag": {
        "type": "categorical",
        "label": "CRRT",
        "options": ["No", "Yes"],
        "default": "No",
    },
    "pneumonia": {
        "type": "categorical",
        "label": "Pneumonia",
        "options": ["No", "Yes"],
        "default": "No",
    },
    "liquid_balance_value": {
        "type": "numeric",
        "label": "Liquid Balance",
        "unit": "mL",
        "min": -11992.08,
        "max": 24421.08,
        "default": 4500.33,
        "step": 1.0,
    },
    "glucose": {
        "type": "numeric",
        "label": "Glucose",
        "unit": "mg/dL",
        "min": 35.0,
        "max": 1027.0,
        "default": 129.0,
        "step": 1.0,
    },
    "albumin": {
        "type": "numeric",
        "label": "Albumin",
        "unit": "g/dL",
        "min": 1.3,
        "max": 5.2,
        "default": 2.9639,
        "step": 0.1,
    },
    "pao2fio2ratio": {
        "type": "numeric",
        "label": "PaO2/FiO2 Ratio",
        "unit": "mmHg",
        "min": 26.0,
        "max": 1327.5,
        "default": 202.3909,
        "step": 1.0,
    },
    "wbc": {
        "type": "numeric",
        "label": "White Blood Cell Count",
        "unit": "10^9/L",
        "min": 0.1,
        "max": 77.8,
        "default": 13.6,
        "step": 0.1,
    },
}
FEATURE_ORDER = list(FEATURE_SPECS.keys())


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(10, 131, 110, 0.18), transparent 26%),
                radial-gradient(circle at 100% 0%, rgba(1, 74, 148, 0.16), transparent 24%),
                linear-gradient(180deg, #f5f8fb 0%, #eaf0f5 100%);
            color: #173042;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card, .panel-card, .metric-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(23, 48, 66, 0.08);
            border-radius: 22px;
            box-shadow: 0 18px 40px rgba(20, 53, 80, 0.08);
            backdrop-filter: blur(8px);
        }
        .hero-card {
            padding: 26px 28px;
            margin-bottom: 18px;
        }
        .panel-card {
            padding: 18px 20px;
            margin-bottom: 16px;
        }
        .metric-card {
            padding: 18px 20px;
            margin-bottom: 12px;
            min-height: 124px;
        }
        .eyebrow {
            color: #0f7a64;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.6rem;
        }
        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1.06;
            color: #143550;
            margin-bottom: 0.65rem;
        }
        .hero-subtitle {
            font-size: 1.02rem;
            line-height: 1.7;
            color: #4d6779;
            max-width: 900px;
            margin: 0;
        }
        .outcome-chip {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .chip-ards {
            background: rgba(209, 55, 84, 0.14);
            color: #b01d3c;
        }
        .chip-no-ards {
            background: rgba(9, 128, 99, 0.14);
            color: #0a7d62;
        }
        .metric-label {
            color: #617a8b;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: #143550;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.35rem;
        }
        .metric-note {
            color: #587284;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 800;
            color: #143550;
            margin-bottom: 0.9rem;
        }
        .small-note {
            color: #5b7383;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    if "model_pipeline" not in artifact:
        raise KeyError("The PKL file does not contain 'model_pipeline'.")
    return artifact


def get_probabilities(artifact: dict[str, Any], input_df: pd.DataFrame) -> tuple[dict[str, float], str]:
    model = artifact["model_pipeline"]
    classes = list(model.named_steps["model"].classes_)
    probabilities = model.predict_proba(input_df)[0]
    probability_map = {label: float(prob) for label, prob in zip(classes, probabilities)}

    threshold_no_ards = float(artifact["threshold_for_positive_label"])
    prob_no_ards = probability_map.get("No", 0.0)
    predicted_class = "No" if prob_no_ards >= threshold_no_ards else "Yes"
    return probability_map, predicted_class


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">ARDS Risk Screening with Elastic-Net Logistic Regression</div>
            <p class="hero-subtitle">
                Enter the current patient variables below to generate an instant prediction.
                The page returns the estimated probabilities for <strong>ARDS</strong> and
                <strong>non-ARDS</strong>, together with the final class decision from the saved model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    st.sidebar.markdown("## Prediction Guide")
    st.sidebar.markdown(
        """
        - Fill in all patient variables.
        - Click `Run Prediction`.
        - Review the ARDS and non-ARDS probabilities.
        - Read the final predicted class.
        """
    )
    st.sidebar.info(
        "This page is designed for single-patient prediction. "
        "It returns the predicted class together with ARDS and non-ARDS probabilities."
    )


def build_input_form() -> tuple[bool, dict[str, Any]]:
    user_inputs: dict[str, Any] = {}

    st.markdown("### Patient Inputs")
    st.caption("All labels are in English. Default values are prefilled for quick single-patient prediction.")

    with st.form("ards_prediction_form", clear_on_submit=False):
        columns = st.columns(2)
        for idx, (feature, spec) in enumerate(FEATURE_SPECS.items()):
            with columns[idx % 2]:
                if spec["type"] == "categorical":
                    default_index = spec["options"].index(spec["default"])
                    user_inputs[feature] = st.selectbox(
                        spec["label"],
                        spec["options"],
                        index=default_index,
                        key=f"field_{feature}",
                    )
                else:
                    label = spec["label"]
                    if spec["unit"]:
                        label = f"{label} ({spec['unit']})"
                    user_inputs[feature] = st.number_input(
                        label,
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=float(spec["default"]),
                        step=float(spec["step"]),
                        key=f"field_{feature}",
                    )

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    return submitted, user_inputs


def render_prediction(probability_map: dict[str, float], predicted_class: str, artifact: dict[str, Any]) -> None:
    prob_ards = probability_map.get("Yes", 0.0)
    prob_no_ards = probability_map.get("No", 0.0)

    outcome_name = "Predicted class: No ARDS" if predicted_class == "No" else "Predicted class: ARDS"
    chip_class = "chip-no-ards" if predicted_class == "No" else "chip-ards"
    chip_text = "The current input pattern is more consistent with a non-ARDS prediction." if predicted_class == "No" else "The current input pattern is more consistent with an ARDS prediction."

    st.markdown("### Prediction Result")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="outcome-chip {chip_class}">{outcome_name}</div>
                <div class="metric-value">{prob_ards:.1%}</div>
                <div class="metric-label">Probability of ARDS</div>
                <div class="metric-note">{chip_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Probability of Non-ARDS</div>
                <div class="metric-value">{prob_no_ards:.1%}</div>
                <div class="metric-note">This is the model-estimated probability for the non-ARDS class.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_resource
def build_shap_explainer() -> Any:
    if shap is None:
        return None

    artifact = load_artifact()
    model_pipeline = artifact["model_pipeline"]
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["model"]

    background_df = pd.DataFrame(
        [[FEATURE_SPECS[col]["default"] for col in FEATURE_ORDER]],
        columns=FEATURE_ORDER,
    )
    background_matrix = preprocessor.transform(background_df)

    return {
        "explainer": shap.LinearExplainer(
            classifier,
            background_matrix,
            feature_names=preprocessor.get_feature_names_out(),
        ),
        "preprocessor": preprocessor,
        "feature_names": list(preprocessor.get_feature_names_out()),
    }


def render_shap_force_plot(input_df: pd.DataFrame) -> None:
    st.markdown("### SHAP Force Plot")

    if shap is None:
        st.info("SHAP is not installed in the current environment.")
        return

    try:
        shap_payload = build_shap_explainer()
        preprocessor = shap_payload["preprocessor"]
        explainer = shap_payload["explainer"]
        feature_names = shap_payload["feature_names"]

        transformed_row = preprocessor.transform(input_df)
        shap_values = explainer.shap_values(transformed_row)
        shap_row = shap_values[0] if hasattr(shap_values, "__len__") else shap_values

        expected_value = explainer.expected_value
        if hasattr(expected_value, "__len__"):
            expected_value = expected_value[0]

        force_plot = shap.force_plot(
            float(expected_value),
            shap_row,
            transformed_row[0],
            feature_names=feature_names,
            matplotlib=False,
            show=False,
        )
        html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        components.html(html, height=320, scrolling=True)
    except Exception as exc:
        st.warning(f"SHAP force plot could not be generated: {exc}")


def main() -> None:
    inject_styles()

    try:
        artifact = load_artifact()
    except Exception as exc:
        st.error(f"Startup failed: {exc}")
        return

    render_header()
    render_sidebar()

    submitted, user_inputs = build_input_form()

    if not submitted:
        return

    input_df = pd.DataFrame([[user_inputs[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)

    try:
        probability_map, predicted_class = get_probabilities(artifact, input_df)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    render_prediction(probability_map, predicted_class, artifact)
    render_shap_force_plot(input_df)


if __name__ == "__main__":
    main()
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import shap
except Exception:
    shap = None


st.set_page_config(
    page_title="ARDS Risk Screening",
    page_icon="AI",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "evalresult_enet.pkl"

FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "ventilator_flag": {
        "type": "categorical",
        "label": "Mechanical Ventilation",
        "options": ["No", "Yes"],
        "default": "Yes",
    },
    "crrt_flag": {
        "type": "categorical",
        "label": "CRRT",
        "options": ["No", "Yes"],
        "default": "No",
    },
    "pneumonia": {
        "type": "categorical",
        "label": "Pneumonia",
        "options": ["No", "Yes"],
        "default": "No",
    },
    "liquid_balance_value": {
        "type": "numeric",
        "label": "Liquid Balance",
        "unit": "mL",
        "min": -11992.08,
        "max": 24421.08,
        "default": 4500.33,
        "step": 1.0,
    },
    "glucose": {
        "type": "numeric",
        "label": "Glucose",
        "unit": "mg/dL",
        "min": 35.0,
        "max": 1027.0,
        "default": 129.0,
        "step": 1.0,
    },
    "albumin": {
        "type": "numeric",
        "label": "Albumin",
        "unit": "g/dL",
        "min": 1.3,
        "max": 5.2,
        "default": 2.9639,
        "step": 0.1,
    },
    "pao2fio2ratio": {
        "type": "numeric",
        "label": "PaO2/FiO2 Ratio",
        "unit": "mmHg",
        "min": 26.0,
        "max": 1327.5,
        "default": 202.3909,
        "step": 1.0,
    },
    "wbc": {
        "type": "numeric",
        "label": "White Blood Cell Count",
        "unit": "10^9/L",
        "min": 0.1,
        "max": 77.8,
        "default": 13.6,
        "step": 0.1,
    },
}
FEATURE_ORDER = list(FEATURE_SPECS.keys())


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(10, 131, 110, 0.18), transparent 26%),
                radial-gradient(circle at 100% 0%, rgba(1, 74, 148, 0.16), transparent 24%),
                linear-gradient(180deg, #f5f8fb 0%, #eaf0f5 100%);
            color: #173042;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card, .panel-card, .metric-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(23, 48, 66, 0.08);
            border-radius: 22px;
            box-shadow: 0 18px 40px rgba(20, 53, 80, 0.08);
            backdrop-filter: blur(8px);
        }
        .hero-card {
            padding: 26px 28px;
            margin-bottom: 18px;
        }
        .panel-card {
            padding: 18px 20px;
            margin-bottom: 16px;
        }
        .metric-card {
            padding: 18px 20px;
            margin-bottom: 12px;
            min-height: 124px;
        }
        .eyebrow {
            color: #0f7a64;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.6rem;
        }
        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1.06;
            color: #143550;
            margin-bottom: 0.65rem;
        }
        .hero-subtitle {
            font-size: 1.02rem;
            line-height: 1.7;
            color: #4d6779;
            max-width: 900px;
            margin: 0;
        }
        .outcome-chip {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .chip-ards {
            background: rgba(209, 55, 84, 0.14);
            color: #b01d3c;
        }
        .chip-no-ards {
            background: rgba(9, 128, 99, 0.14);
            color: #0a7d62;
        }
        .metric-label {
            color: #617a8b;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: #143550;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.35rem;
        }
        .metric-note {
            color: #587284;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 800;
            color: #143550;
            margin-bottom: 0.9rem;
        }
        .small-note {
            color: #5b7383;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    if "model_pipeline" not in artifact:
        raise KeyError("The PKL file does not contain 'model_pipeline'.")
    return artifact


def get_probabilities(artifact: dict[str, Any], input_df: pd.DataFrame) -> tuple[dict[str, float], str]:
    model = artifact["model_pipeline"]
    classes = list(model.named_steps["model"].classes_)
    probabilities = model.predict_proba(input_df)[0]
    probability_map = {label: float(prob) for label, prob in zip(classes, probabilities)}

    threshold_no_ards = float(artifact["threshold_for_positive_label"])
    prob_no_ards = probability_map.get("No", 0.0)
    predicted_class = "No" if prob_no_ards >= threshold_no_ards else "Yes"
    return probability_map, predicted_class


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">ARDS Risk Screening with Elastic-Net Logistic Regression</div>
            <p class="hero-subtitle">
                Enter the current patient variables below to generate an instant prediction.
                The page returns the estimated probabilities for <strong>ARDS</strong> and
                <strong>non-ARDS</strong>, together with the final class decision from the saved model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    st.sidebar.markdown("## Prediction Guide")
    st.sidebar.markdown(
        """
        - Fill in all patient variables.
        - Click `Run Prediction`.
        - Review the ARDS and non-ARDS probabilities.
        - Read the final predicted class.
        """
    )
    st.sidebar.info(
        "This page is designed for single-patient prediction. "
        "It returns the predicted class together with ARDS and non-ARDS probabilities."
    )


def build_input_form() -> tuple[bool, dict[str, Any]]:
    user_inputs: dict[str, Any] = {}

    st.markdown("### Patient Inputs")
    st.caption("All labels are in English. Default values are prefilled for quick single-patient prediction.")

    with st.form("ards_prediction_form", clear_on_submit=False):
        columns = st.columns(2)
        for idx, (feature, spec) in enumerate(FEATURE_SPECS.items()):
            with columns[idx % 2]:
                if spec["type"] == "categorical":
                    default_index = spec["options"].index(spec["default"])
                    user_inputs[feature] = st.selectbox(
                        spec["label"],
                        spec["options"],
                        index=default_index,
                        key=f"field_{feature}",
                    )
                else:
                    label = spec["label"]
                    if spec["unit"]:
                        label = f"{label} ({spec['unit']})"
                    user_inputs[feature] = st.number_input(
                        label,
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=float(spec["default"]),
                        step=float(spec["step"]),
                        key=f"field_{feature}",
                    )

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    return submitted, user_inputs


def render_prediction(probability_map: dict[str, float], predicted_class: str, artifact: dict[str, Any]) -> None:
    prob_ards = probability_map.get("Yes", 0.0)
    prob_no_ards = probability_map.get("No", 0.0)

    outcome_name = "Predicted class: No ARDS" if predicted_class == "No" else "Predicted class: ARDS"
    chip_class = "chip-no-ards" if predicted_class == "No" else "chip-ards"
    chip_text = "The current input pattern is more consistent with a non-ARDS prediction." if predicted_class == "No" else "The current input pattern is more consistent with an ARDS prediction."

    st.markdown("### Prediction Result")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="outcome-chip {chip_class}">{outcome_name}</div>
                <div class="metric-value">{prob_ards:.1%}</div>
                <div class="metric-label">Probability of ARDS</div>
                <div class="metric-note">{chip_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Probability of Non-ARDS</div>
                <div class="metric-value">{prob_no_ards:.1%}</div>
                <div class="metric-note">This is the model-estimated probability for the non-ARDS class.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_resource
def build_shap_explainer() -> Any:
    if shap is None:
        return None

    artifact = load_artifact()
    model_pipeline = artifact["model_pipeline"]
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["model"]

    background_df = pd.DataFrame(
        [[FEATURE_SPECS[col]["default"] for col in FEATURE_ORDER]],
        columns=FEATURE_ORDER,
    )
    background_matrix = preprocessor.transform(background_df)

    return {
        "explainer": shap.LinearExplainer(
            classifier,
            background_matrix,
            feature_names=preprocessor.get_feature_names_out(),
        ),
        "preprocessor": preprocessor,
        "feature_names": list(preprocessor.get_feature_names_out()),
    }


def render_shap_force_plot(input_df: pd.DataFrame) -> None:
    st.markdown("### SHAP Force Plot")

    if shap is None:
        st.info("SHAP is not installed in the current environment.")
        return

    try:
        shap_payload = build_shap_explainer()
        preprocessor = shap_payload["preprocessor"]
        explainer = shap_payload["explainer"]
        feature_names = shap_payload["feature_names"]

        transformed_row = preprocessor.transform(input_df)
        shap_values = explainer.shap_values(transformed_row)
        shap_row = shap_values[0] if hasattr(shap_values, "__len__") else shap_values

        expected_value = explainer.expected_value
        if hasattr(expected_value, "__len__"):
            expected_value = expected_value[0]

        force_plot = shap.force_plot(
            float(expected_value),
            shap_row,
            transformed_row[0],
            feature_names=feature_names,
            matplotlib=False,
            show=False,
        )
        html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        components.html(html, height=320, scrolling=True)
    except Exception as exc:
        st.warning(f"SHAP force plot could not be generated: {exc}")


def main() -> None:
    inject_styles()

    try:
        artifact = load_artifact()
    except Exception as exc:
        st.error(f"Startup failed: {exc}")
        return

    render_header()
    render_sidebar()

    submitted, user_inputs = build_input_form()

    if not submitted:
        return

    input_df = pd.DataFrame([[user_inputs[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)

    try:
        probability_map, predicted_class = get_probabilities(artifact, input_df)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    render_prediction(probability_map, predicted_class, artifact)
    render_shap_force_plot(input_df)


if __name__ == "__main__":
    main()
