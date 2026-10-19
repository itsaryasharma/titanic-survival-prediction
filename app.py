"""Titanic Survival Prediction — Streamlit application.

Run with:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import CLASS_NAMES, DEFAULT_THRESHOLD, EXCLUDED_COLUMNS, PORTS
from src.data import load_titanic
from src.features import CHILD_MAX_AGE, family_type
from src.predict import (
    ensure_trained,
    explain_passenger,
    load_metrics,
    load_model,
    predict_survival,
    range_warnings,
    validate_passenger,
    what_if,
)

BLUE, RED, GREEN, GREY = "#2563EB", "#DC2626", "#059669", "#9CA3AF"
PLOT_MARGIN = dict(l=10, r=10, t=10, b=10)
ASSETS = Path(__file__).resolve().parent / "assets"
FEATURE_LABELS = {"sex & class": "Sex & class", "age": "Age", "family": "Family aboard",
                  "fare": "Fare", "embarked": "Port of embarkation"}

PRESETS = {  # illustrative profiles (not specific historical people)
    "1st-class woman, 38": dict(pclass=1, sex="female", age=38.0, sibsp=1, parch=0, embarked="C"),
    "3rd-class man, 25, alone": dict(pclass=3, sex="male", age=25.0, sibsp=0, parch=0, embarked="S"),
    "3rd-class boy, 6, large family": dict(pclass=3, sex="male", age=6.0, sibsp=3, parch=2, embarked="S"),
    "2nd-class man, 45": dict(pclass=2, sex="male", age=45.0, sibsp=1, parch=1, embarked="S"),
}
DEFAULT = PRESETS["3rd-class man, 25, alone"]


@st.cache_resource(show_spinner="Preparing the model…")
def get_model_and_metrics():
    ensure_trained()
    return load_model(), load_metrics()


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_titanic()


def apply_preset(name: str) -> None:
    for k, v in PRESETS[name].items():
        st.session_state[k] = v
    st.session_state["age_unknown"] = False
    st.session_state["typical_fare"] = True


# --------------------------------------------------------------------------- #
# Predict tab
# --------------------------------------------------------------------------- #
def render_form(metrics: dict) -> dict:
    st.subheader("Passenger details")
    cols = st.columns(2)
    for i, name in enumerate(PRESETS):
        cols[i % 2].button(name, on_click=apply_preset, args=(name,), width="stretch", key=f"preset_{i}")

    pclass = st.radio("Ticket class", [1, 2, 3], format_func=CLASS_NAMES.get, horizontal=True, key="pclass")
    sex = st.radio("Sex (as recorded in 1912)", ["female", "male"], horizontal=True, key="sex")
    a1, a2 = st.columns([2, 1])
    age_unknown = a2.checkbox("Age unknown", key="age_unknown",
                              help="20% of real passengers have no recorded age; the model can handle this.")
    age = a1.number_input("Age (years)", 0.0, 100.0, step=1.0, key="age", disabled=age_unknown)
    f1, f2 = st.columns(2)
    sibsp = f1.number_input("Siblings / spouses aboard", 0, 10, step=1, key="sibsp")
    parch = f2.number_input("Parents / children aboard", 0, 10, step=1, key="parch")
    embarked = st.selectbox("Port of embarkation", list(PORTS), format_func=lambda k: f"{PORTS[k]} ({k})",
                            key="embarked")
    typical = st.checkbox("Use the typical (median) fare for this class", key="typical_fare")
    median_fare = metrics["train_stats"]["median_fare_by_class"][str(pclass)]
    if typical:
        fare = median_fare
        st.caption(f"Fare: £{fare:.2f} (median {CLASS_NAMES[pclass]} fare in the training data, 1912 pounds)")
    else:
        fare = st.number_input("Fare paid (1912 pounds)", 0.0, 1000.0, value=float(round(median_fare, 2)),
                               step=1.0, key="fare")
    return dict(pclass=pclass, sex=sex, age=None if age_unknown else age, sibsp=int(sibsp),
                parch=int(parch), fare=float(fare), embarked=embarked)


def render_prediction(model, metrics: dict, p: dict) -> None:
    prob = predict_survival(model, p)
    survived = prob >= DEFAULT_THRESHOLD
    ui.result_card(
        "Estimated survival probability", f"{prob:.0%}",
        ("The model leans towards survival" if survived else "The model leans towards not surviving")
        + " for a passenger with this profile in 1912.",
        GREEN if survived else RED,
    )
    fig = go.Figure(go.Bar(x=[prob], y=[""], orientation="h", marker_color=GREEN if survived else RED,
                           text=[f"{prob:.0%}"], textposition="inside", width=0.5,
                           insidetextfont=dict(color="white")))
    fig.add_vline(x=metrics["dataset"]["survival_rate"], line_dash="dot", line_color="#64748B",
                  annotation_text=f"overall survival rate {metrics['dataset']['survival_rate']:.0%}",
                  annotation_position="top")
    fig.update_layout(height=110, margin=dict(l=10, r=10, t=28, b=10), plot_bgcolor="rgba(128,128,128,0.15)",
                      xaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f":material/info: **How to read {prob:.0%}:** if there had been 100 passengers with exactly this profile, "
        f"the model would expect about {round(prob * 100)} of them to survive. It is an estimate for the "
        "*profile*, learned from 1912 records, not a statement about what happened to any one person. The "
        "verdict wording simply compares the estimate with 50%."
    )
    for w in range_warnings(p, metrics["train_stats"]):
        st.warning(f"**Outside the training data:** {w} The estimate is an extrapolation.")
    if p["age"] is None:
        st.info("Age unknown: the model uses the median training age plus an 'age missing' signal.")


def render_explanations(model, metrics: dict, p: dict) -> None:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Why? Contribution of each attribute**")
        if metrics["explanation"] is None:
            st.info("Per-feature contributions are only available for the logistic-regression model.")
        else:
            exp = explain_passenger(model, p, metrics["explanation"])
            items = list(exp["contributions"].items())
            fig = go.Figure(go.Bar(
                x=[v for _, v in items], y=[FEATURE_LABELS[k] for k, _ in items], orientation="h",
                marker_color=[GREEN if v > 0 else RED for _, v in items],
                text=[f"{v:+.2f}" for _, v in items], textposition="outside",
            ))
            span = max(abs(v) for _, v in items) * 1.35 + 0.1
            fig.update_layout(height=300, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(range=[-span, span], title="Change in log-odds vs. reference passenger"))
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"Compared with an *average* training passenger (estimated survival "
                f"{exp['reference_probability']:.0%}). Green bars raise the estimate, red bars lower it. "
                "Bars add up exactly to the passenger's score, because the model is linear in log-odds. "
                "These are **associations in the 1912 data, not causes**: for example, the port effect mostly "
                "reflects who boarded where (many Cherbourg passengers travelled 1st class)."
            )
    with right:
        st.markdown("**What if…? Same passenger, one attribute changed**")
        by_class = what_if(model, p, "pclass", [1, 2, 3])
        by_sex = what_if(model, p, "sex", ["female", "male"])
        frame = pd.DataFrame(
            [{"scenario": CLASS_NAMES[k], "group": "Ticket class", "p": v} for k, v in by_class.items()]
            + [{"scenario": k, "group": "Sex", "p": v} for k, v in by_sex.items()]
        )
        current = {CLASS_NAMES[p["pclass"]], p["sex"]}
        frame["is_current"] = frame["scenario"].isin(current)
        fig = px.bar(frame, x="scenario", y="p", facet_col="group", text=frame["p"].map("{:.0%}".format),
                     color="is_current", color_discrete_map={True: BLUE, False: GREY})
        fig.update_xaxes(matches=None, title=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                          plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[0, 1.05], tickformat=".0%", title="Survival"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Blue bars are this passenger's actual values. When the fare is set to 'typical', the "
                   "class comparison keeps the fare fixed, so it isolates the effect of class itself.")


def render_predict(model, metrics: dict) -> None:
    st.markdown(
        "⚓ **A historical-data exercise.** This model estimates how passengers with a given profile fared "
        "in 1912, from the recorded outcomes of 1,309 real passengers. It does not determine what happened to "
        "any particular person, and it reflects the social rules of that night (\"women and children first\", "
        "class-based access to lifeboats), not a fair or universal judgement."
    )
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        p = render_form(metrics)
    with right:
        st.subheader("Estimate")
        errors = validate_passenger(p)
        if errors:
            for e in errors:
                st.error(e)
            return
        render_prediction(model, metrics, p)
    st.divider()
    render_explanations(model, metrics, p)


# --------------------------------------------------------------------------- #
# Model card tab
# --------------------------------------------------------------------------- #
def render_model_card(metrics: dict) -> None:
    t, ds = metrics["test"], metrics["dataset"]
    st.subheader("How well does the model perform?")
    st.caption(
        f"Evaluated once on {ds['n_test']} held-out passengers (stratified 80/20 split, "
        f"random_state={ds['random_state']}). All numbers are produced by `src/train.py` and read from "
        f"`models/metrics.json`. Trained {metrics['trained_at_utc']} with scikit-learn {metrics['sklearn_version']}."
    )
    m = st.columns(5)
    m[0].metric("Accuracy", f"{t['accuracy']:.1%}", border=True)
    m[1].metric("Precision", f"{t['precision']:.3f}", help="Of passengers predicted to survive, the share who did.", border=True)
    m[2].metric("Recall", f"{t['recall']:.3f}", help="Of passengers who survived, the share the model found.", border=True)
    m[3].metric("F1", f"{t['f1']:.3f}", border=True)
    m[4].metric("ROC-AUC", f"{t['roc_auc']:.3f}", border=True)

    st.markdown("#### Is the model better than simple rules?")
    rows = [{"Approach": f"{metrics['selected_model']} (this app)", "Accuracy": t["accuracy"], "F1": t["f1"]}]
    rows += [{"Approach": name, "Accuracy": b["accuracy"], "F1": b["f1"]} for name, b in metrics["baselines"].items()]
    frame = pd.DataFrame(rows)
    fig = px.bar(frame.melt(id_vars="Approach", var_name="metric"), x="value", y="Approach", color="metric",
                 barmode="group", orientation="h", text_auto=".3f",
                 color_discrete_map={"Accuracy": BLUE, "F1": "#93C5FD"})
    fig.update_layout(height=260, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)", legend_title=None,
                      xaxis=dict(range=[0, 1], title="Score on the 262 test passengers (higher is better)"), yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    gain = t["accuracy"] - metrics["baselines"]["Women survive, men do not"]["accuracy"]
    sig = metrics["baseline_significance"]
    n_test = metrics["dataset"]["n_test"]
    verdict = ("This difference **is** statistically significant" if sig["p_value"] < 0.05 else
               "On this test set the difference is **not statistically significant**, so treat the two as "
               "having roughly equal accuracy")
    st.info(
        f"**Honest reading.** The one-line rule *women survive, men do not* is a strong baseline. The model is "
        f"{gain * 100:+.1f} percentage points more accurate on the {n_test} test passengers (about "
        f"{round(gain * n_test)} passengers). {verdict} (exact McNemar test, p = {sig['p_value']:.2f}: the model "
        f"was right where the rule was wrong for {sig['model_right_rule_wrong']} passengers, and the reverse for "
        f"{sig['rule_right_model_wrong']}). The model's real advantages are graded probabilities instead of "
        "yes/no answers (ROC-AUC measures how well it ranks passengers), per-passenger explanations, and "
        "handling cases the rule ignores, such as boys or women in 3rd class."
    )

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Confusion matrix (test set, threshold 50%)**")
        fig = px.imshow(t["confusion_matrix"], x=["did not survive", "survived"], y=["did not survive", "survived"],
                        text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Passengers"))
        fig.update_layout(height=300, margin=PLOT_MARGIN, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown(f"**ROC curve (AUC {t['roc_auc']:.3f})**")
        fig = go.Figure(go.Scatter(x=t["roc_curve"]["x"], y=t["roc_curve"]["y"], mode="lines",
                                   line=dict(color=BLUE, width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=GREY)))
        fig.update_layout(height=300, margin=PLOT_MARGIN, showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, width="stretch")

    c, d = st.columns(2, gap="large")
    with c:
        st.markdown(f"**Model comparison ({metrics['cv_folds']}-fold CV, training set)**")
        cv = pd.DataFrame(metrics["cv_results"]).T.rename(columns={"accuracy": "Accuracy", "accuracy_std": "± std", "roc_auc": "ROC-AUC", "roc_auc_std": "± std "})
        cv.index.name = "Model"
        st.dataframe(cv.style.format("{:.4f}"), width="stretch")
        st.caption(f"Selected: **{metrics['selected_model']}**. {metrics['selection_rule']}")
    with d:
        leak = metrics["leakage_experiment"]
        st.markdown("**Data-leakage experiment**")
        st.metric("CV accuracy, honest features", f"{leak['honest_cv']['accuracy']:.1%}", border=True)
        st.metric("CV accuracy if we (wrongly) add the lifeboat column", f"{leak['with_boat_cv']['accuracy']:.1%}",
                  delta=f"{leak['with_boat_cv']['accuracy'] - leak['honest_cv']['accuracy']:+.1%} (fake)",
                  delta_color="off", border=True)
        st.caption("Whether someone got into a lifeboat is only known *after* the outcome. Using it would make "
                   "the model look far better while being useless for genuine prediction. It is excluded.")

    if metrics["explanation"]:
        st.markdown("**Learned weights (logistic regression, on encoded features)**")
        e = metrics["explanation"]
        frame = pd.DataFrame({"feature": e["feature_names"], "weight": e["coef"]}).sort_values("weight")
        frame["feature"] = frame["feature"].str.split("__").str[1]
        fig = px.bar(frame, x="weight", y="feature", orientation="h", color=frame["weight"] > 0,
                     color_discrete_map={True: GREEN, False: RED})
        fig.update_layout(height=480, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                          xaxis_title="Weight (positive = higher survival odds)", yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("Numeric features (age, log fare) are standardised, so their weights are per standard deviation.")


# --------------------------------------------------------------------------- #
# How it works tab
# --------------------------------------------------------------------------- #
def render_how_it_works(metrics: dict) -> None:
    st.subheader("From passenger details to a probability")
    st.markdown(
        f"""
1. **Inputs**: ticket class, sex, age (or unknown), siblings/spouses and parents/children aboard, fare, port.
2. **Feature engineering** (inside the saved pipeline):
   - `family_type`: alone / small (2–4 people) / large (5+), from 1 + siblings/spouses + parents/children.
   - `sex_class`: combined category such as *female_3*. This lets a linear model learn that being female
     helped far less in 3rd class.
   - `is_child`: age ≤ {CHILD_MAX_AGE} ("women **and children** first").
   - `log_fare = log(1 + fare)`: fares are heavily skewed (most under £30, a few over £500).
3. **Missing values**: age → training median **plus** an "age was missing" indicator; port → most common port.
4. **Encoding & scaling**: categories → one-hot columns; numbers → standardised (fitted on training data only).
5. **{metrics['selected_model']}** → probability of survival.
        """
    )
    st.latex(r"P(\text{survived}\mid x) = \sigma\Big(b + \sum_j w_j x_j\Big) = \frac{1}{1+e^{-(b+\sum_j w_j x_j)}}")
    st.markdown("#### Columns deliberately **not** used")
    st.dataframe(pd.DataFrame([{"Column": k, "Reason": v} for k, v in EXCLUDED_COLUMNS.items()]),
                 width="stretch", hide_index=True)
    st.markdown("#### What this model can and cannot tell you")
    st.markdown(
        f"""
- ✅ How survival chances in the 1912 data varied with class, sex, age and family situation.
- ⚠️ It **cannot** know where a passenger was when the ship struck the iceberg, whether they reached the
  boat deck, crew decisions, or luck. These factors decided many real outcomes.
- ⚠️ Trained on {metrics['dataset']['n_train']:,} passengers; the crew is not included. Estimates for
  unusual profiles (e.g. very large families) rest on very few examples.
- ⚠️ The patterns reflect 1912 social norms and inequality. They must not be generalised to other
  disasters or used to judge people.
        """
    )


# --------------------------------------------------------------------------- #
# Data insights tab
# --------------------------------------------------------------------------- #
def render_insights(df: pd.DataFrame, metrics: dict) -> None:
    st.image(str(ASSETS / "rms_titanic_1912.jpg"), width=900,
             caption="RMS Titanic leaving Southampton, 10 April 1912. Photo: F. G. O. Stuart, public domain "
                     "(Wikimedia Commons; see assets/ATTRIBUTIONS.md).")
    st.subheader("What the data shows")
    st.caption(f"All {len(df):,} passengers in the titanic3 dataset (training and test). Overall survival "
               f"rate: {df['survived'].mean():.1%}.")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Survival rate by sex and class**")
        g = df.groupby(["pclass", "sex"])["survived"].agg(["mean", "size"]).reset_index()
        g["class"] = g["pclass"].map(CLASS_NAMES)
        fig = px.bar(g, x="class", y="mean", color="sex", barmode="group", text=g["mean"].map("{:.0%}".format),
                     hover_data={"size": True}, color_discrete_map={"female": "#DB2777", "male": BLUE})
        fig.update_layout(height=320, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)", legend_title=None,
                          yaxis=dict(tickformat=".0%", title="Survival rate", range=[0, 1.05]), xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Survival rate by age group and sex**")
        bins = [0, CHILD_MAX_AGE, 19, 30, 45, 60, 100]
        labels = ["0–12", "13–19", "20–30", "31–45", "46–60", "60+"]
        d = df.assign(age_group=pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True).astype(str))
        d.loc[df["age"].isna(), "age_group"] = "unknown"
        g = d.groupby(["age_group", "sex"])["survived"].agg(["mean", "size"]).reset_index()
        fig = px.bar(g, x="age_group", y="mean", color="sex", barmode="group", hover_data={"size": True},
                     category_orders={"age_group": labels + ["unknown"]},
                     color_discrete_map={"female": "#DB2777", "male": BLUE})
        fig.update_layout(height=320, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)", legend_title=None,
                          yaxis=dict(tickformat=".0%", title="Survival rate", range=[0, 1.05]), xaxis_title="Age")
        st.plotly_chart(fig, width="stretch")
    c, e = st.columns(2, gap="large")
    with c:
        st.markdown("**Survival rate by family size**")
        d = df.assign(family=family_type(df["sibsp"] + df["parch"] + 1))
        g = d.groupby("family")["survived"].agg(["mean", "size"]).reindex(["alone", "small", "large"]).reset_index()
        fig = px.bar(g, x="family", y="mean", text=g.apply(lambda r: f"{r['mean']:.0%} (n={r['size']})", axis=1),
                     color_discrete_sequence=[BLUE])
        fig.update_layout(height=300, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)",
                          yaxis=dict(tickformat=".0%", title="Survival rate", range=[0, 1.05]), xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with e:
        st.markdown("**Missing values in the raw data**")
        miss = pd.DataFrame({"column": list(metrics["dataset"]["missing"]),
                             "missing": list(metrics["dataset"]["missing"].values())})
        miss["share"] = (miss["missing"] / len(df)).map("{:.1%}".format)
        st.dataframe(miss, width="stretch", hide_index=True)
        st.caption("Cabin, boat, body and home.dest are not used (see *How it works*). Age, fare and port are "
                   "imputed inside the model pipeline using training-set statistics only.")
    st.markdown(
        "**Source:** `titanic3` dataset, Vanderbilt University Department of Biostatistics "
        "(compiled by Thomas Cason from the *Encyclopedia Titanica*). Data obtained from "
        "http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics."
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="wide")
    ui.inject_css()
    try:
        model, metrics = get_model_and_metrics()
        df = get_data()
    except Exception as exc:
        st.error(f"The model or data could not be loaded: {exc}")
        st.info("Run `python -m src.train` from the project folder, then reload this page.")
        st.stop()

    for k, v in DEFAULT.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("age_unknown", False)
    st.session_state.setdefault("typical_fare", True)

    t = metrics["test"]
    ui.hero(
        "Titanic Survival Prediction",
        "Describe a passenger to estimate their chance of surviving the 1912 disaster, and see which "
        "factors drive the estimate.",
        [metrics["selected_model"], f"Accuracy {t['accuracy']:.1%} · ROC-AUC {t['roc_auc']:.3f}",
         f"{metrics['dataset']['n_passengers']:,} real passengers"],
    )
    tab_p, tab_m, tab_h, tab_i = st.tabs(["🚢 Predict", "📊 Model card", "🧠 How it works", "📈 Data insights"])
    with tab_p:
        render_predict(model, metrics)
    with tab_m:
        render_model_card(metrics)
    with tab_h:
        render_how_it_works(metrics)
    with tab_i:
        render_insights(df, metrics)
    ui.footer("Titanic Survival Prediction · Arya Kumar · Educational machine-learning project")


main()
