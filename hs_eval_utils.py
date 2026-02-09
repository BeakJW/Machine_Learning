from pandas import DataFrame, merge
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np

from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

import shap

my_dpi = 100


def hs_get_scores(estimator, x_test, y_true):
    """모델 성능 평가 함수"""
    if hasattr(estimator, "named_steps"):
        classname = estimator.named_steps["model"].__class__.__name__
    else:
        classname = estimator.__class__.__name__

    y_pred = estimator.predict(x_test)

    return DataFrame(
        {
            "결정계수(R2)": r2_score(y_true, y_pred),
            "평균절대오차(MAE)": mean_absolute_error(y_true, y_pred),
            "평균제곱오차(MSE)": mean_squared_error(y_true, y_pred),
            "평균오차(RMSE)": np.sqrt(mean_squared_error(y_true, y_pred)),
            "평균 절대 백분오차 비율(MAPE)": mean_absolute_percentage_error(
                y_true, y_pred
            ),
            "평균 비율 오차(MPE)": np.mean((y_true - y_pred) / y_true * 100),
        },
        index=[classname],
    )


def hs_learning_cv(
    estimator,
    x,
    y,
    scoring="neg_root_mean_squared_error",
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
):
    """학습 곡선 분석 및 과적합 판정 함수"""
    train_sizes, train_scores, cv_scores = learning_curve(
        estimator=estimator,
        X=x,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        shuffle=True,
        random_state=52,
    )

    if hasattr(estimator, "named_steps"):
        classname = estimator.named_steps["model"].__class__.__name__
    else:
        classname = estimator.__class__.__name__

    # neg RMSE -> RMSE
    train_rmse = -train_scores
    cv_rmse = -cv_scores

    # 평균 / 표준편차
    train_mean = train_rmse.mean(axis=1)
    cv_mean = cv_rmse.mean(axis=1)
    cv_std = cv_rmse.std(axis=1)

    # 마지막 지점 기준 정량 판정
    final_train = train_mean[-1]
    final_cv = cv_mean[-1]
    final_std = cv_std[-1]
    gap_ratio = final_train / final_cv
    var_ratio = final_std / final_cv

    # 과소적합 기준선
    y_mean = y.mean()
    rmse_naive = np.sqrt(np.mean((y - y_mean) ** 2))
    std_y = y.std()
    min_r2 = 0.10
    rmse_r2 = np.sqrt((1 - min_r2) * np.var(y))
    some_threshold = min(rmse_naive, std_y, rmse_r2)

    # 판정 로직
    if gap_ratio >= 0.95 and final_cv > some_threshold:
        status = "⚠️ 과소적합 (bias 큼)"
    elif gap_ratio <= 0.8:
        status = "⚠️ 과대적합 (variance 큼)"
    elif gap_ratio <= 0.95 and var_ratio <= 0.10:
        status = "✅ 일반화 양호"
    elif var_ratio > 0.15:
        status = "⚠️ 데이터 부족 / 분산 큼"
    else:
        status = "⚠️ 판단 유보"

    # 정량 결과 표
    result_df = DataFrame(
        {
            "Train RMSE": [final_train],
            "CV RMSE 평균": [final_cv],
            "CV RMSE 표준편차": [final_std],
            "Train/CV 비율": [gap_ratio],
            "CV 변동성 비율": [var_ratio],
            "판정 결과": [status],
        },
        index=[classname],
    )

    # 학습곡선 시각화
    figsize = (1280 / my_dpi, 720 / my_dpi)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=my_dpi)

    sb.lineplot(
        x=train_sizes,
        y=train_mean,
        marker="o",
        markeredgecolor="#ffffff",
        label="Train RMSE",
        ax=ax,
    )

    sb.lineplot(
        x=train_sizes,
        y=cv_mean,
        marker="o",
        markeredgecolor="#ffffff",
        label="CV RMSE",  # 원본에서 "Train RMSE"로 되어 있던 버그 수정
        ax=ax,
    )

    # x축과 y축이 바뀌어 있던 것 수정
    ax.set_xlabel("훈련 데이터 수", fontsize=8, labelpad=5)
    ax.set_ylabel("RMSE", fontsize=8, labelpad=5)
    ax.set_title("학습곡선 (Learning Curve)", fontsize=12, pad=8)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.close()

    return result_df


def hs_get_score_cv(
    estimator,
    x_test,
    y_test,
    x_origin,
    y_origin,
    scoring="neg_root_mean_squared_error",
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
):
    """성능평가와 과적합 판정을 동시에 수행"""
    score_df = hs_get_scores(estimator, x_test, y_test)
    cv_df = hs_learning_cv(
        estimator,
        x_origin,
        y_origin,
        scoring=scoring,
        cv=cv,
        train_sizes=train_sizes,
        n_jobs=n_jobs,
    )
    return merge(score_df, cv_df, left_index=True, right_index=True)


def feature_importance(model, x_train, y_train):
    """변수 중요도 계산 및 시각화"""
    perm = permutation_importance(
        estimator=model,
        X=x_train,
        y=y_train,
        scoring="r2",
        n_repeats=30,
        random_state=42,
        n_jobs=-1,
    )

    perm_df = DataFrame(
        {
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        },
        index=x_train.columns,
    ).sort_values("importance_mean", ascending=False)

    # 누적 중요도 추가
    perm_df["importance_cumsum"] = perm_df["importance_mean"].cumsum()

    df = perm_df.sort_values(by="importance_mean", ascending=False)

    figsize = (1280 / my_dpi, 600 / my_dpi)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=my_dpi)

    sb.barplot(data=df, x="importance_mean", y=df.index, ax=ax)

    ax.set_title("Permutation Importance", fontsize=12)
    ax.set_xlabel(
        "Permutation Importance (mean)", fontsize=10
    )  # 오타 수정: ermutation -> Permutation
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()

    return perm_df


def hs_shap_analysis(
    model, x: DataFrame, plot: bool = True, width: int = 1600, height: int = 800
):
    """SHAP 분석 일괄 처리 함수"""
    # 1. SHAP Explainer
    explainer = shap.TreeExplainer(model)

    # 2. SHAP 값 계산
    shap_values = explainer.shap_values(x)

    # 3. DataFrame 변환
    shap_df = DataFrame(
        shap_values,
        columns=x.columns,
        index=x.index,
    )

    # 4. 요약 통계
    summary_df = DataFrame(
        {
            "feature": shap_df.columns,
            "mean_abs_shap": shap_df.abs().mean().values,
            "mean_shap": shap_df.mean().values,
            "std_shap": shap_df.std().values,
        }
    )

    # 5. 영향 방향 (보수적 표현)
    summary_df["direction"] = np.where(
        summary_df["mean_shap"] > 0,
        "양(+) 경향",
        np.where(summary_df["mean_shap"] < 0, "음(-) 경향", "혼합/미약"),
    )

    # 6. 변동성 지표
    summary_df["cv"] = summary_df["std_shap"] / (summary_df["mean_abs_shap"] + 1e-9)

    summary_df["variability"] = np.where(
        summary_df["cv"] < 1,
        "stable",  # 변동성 낮음
        "variable",  # 변동성 큼
    )

    # 7. 중요도 기준 정렬
    summary_df = summary_df.sort_values("mean_abs_shap", ascending=False).reset_index(
        drop=True
    )

    # 8. 주요 변수 표시 (누적 80%)
    total_importance = summary_df["mean_abs_shap"].sum()
    summary_df["importance_ratio"] = summary_df["mean_abs_shap"] / total_importance
    summary_df["importance_cumsum"] = summary_df["importance_ratio"].cumsum()

    summary_df["is_important"] = np.where(
        summary_df["importance_cumsum"] <= 0.80,
        "core",
        "secondary",
    )

    # 9. 시각화
    if plot:
        shap.summary_plot(shap_values, x, show=False)

        fig = plt.gcf()
        fig.set_size_inches(width / my_dpi, height / my_dpi)

        plt.title("SHAP Summary Plot", fontsize=10, pad=10)
        plt.xlabel("SHAP value", fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=8)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.close()

    return summary_df, shap_values


def hs_shap_dependence_analysis(
    summary_df: DataFrame,
    shap_values,
    x_train: DataFrame,
    include_secondary: bool = False,
    width: int = 1600,
    height: int = 800,
):
    """SHAP dependence plot 시각화 함수"""
    # 1. 주 대상 변수 (Core + Variable)
    main_features = summary_df[
        (summary_df["is_important"] == "core")
        & (summary_df["variability"] == "variable")
    ]["feature"].tolist()

    # 2. 상호작용 후보 변수
    interaction_features = summary_df[summary_df["is_important"] == "core"][
        "feature"
    ].tolist()

    if include_secondary and len(interaction_features) < 2:
        interaction_features.extend(
            summary_df[summary_df["is_important"] == "secondary"]["feature"].tolist()
        )

    # 3. 변수 쌍 생성 (자기 자신 제외)
    pairs = []
    for f in main_features:
        for inter in interaction_features:
            if f != inter:
                pairs.append((f, inter))

    # 중요도 순 정렬
    importance_rank = {}
    for i, row in summary_df.iterrows():
        importance_rank[row["feature"]] = i

    pairs = sorted(pairs, key=lambda x: importance_rank.get(x[0], 999))

    # 4. dependence plot 일괄 생성
    for feature_name, interaction_name in pairs:
        shap.dependence_plot(
            feature_name,
            shap_values,
            x_train,
            interaction_index=interaction_name,
            show=False,
        )

        fig = plt.gcf()
        fig.set_size_inches(width / my_dpi, height / my_dpi)

        plt.title(
            f"SHAP Dependence Plot: {feature_name} × {interaction_name}",
            fontsize=10,
            pad=10,
        )
        plt.xlabel(feature_name, fontsize=10)
        plt.ylabel(f"SHAP value for {feature_name}", fontsize=10)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=8)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.close()

    return pairs


def hs_feature_importance(model, x_train, y_train):
    if isinstance(model, XGBRegressor):
        booster = model.get_booster()
        imp = booster.get_score(importance_type="gain")
        imp_sr = Series(imp)
        imp_df = DataFrame(imp_sr, columns=["importance"])
    else:
        imp_df = permutation_importance(
            estimator=best_model,
            X=x_train,
            y=y_train,
            scoring="r2",
            n_repeats=30,
            random_state=42,
            n_jobs=-1,
        )

        # 결과 정리
        imp_df = DataFrame(
            {"importance": imp_df.importances_mean}, index=x_train.columns
        )

    # 중요도 비율 + 누적 중요도 계산
    imp_df["ratio"] = imp_df["importance"] / imp_df["importance"].sum()
    imp_df.sort_values("ratio", ascending=False, inplace=True)
    imp_df["cumsum"] = imp_df["ratio"].cumsum()

    # 시각화
    df = imp_df.sort_values(by="ratio", ascending=False)
    threshold = 0.9

    height = len(imp_df) * 60
    figsize = (1280 / 100, height / 100)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=my_dpi)

    sb.barplot(data=df, x="importance", y=df.index)

    # 값 라벨 추가
    for i, v in enumerate(imp_df["importance"]):
        ax.text(
            v + 0.005,  # 막대 끝에서 약간 오른쪽
            i,  # y 위치
            f"{v:.1f} ({imp_df.iloc[i]['cumsum']*100:.1f}%)",  # 표시 형식
            va="center",
        )

    ax.set_title("Feature Importance", fontsize=9)
    ax.set_xlabel("importance(cumsum)", fontsize=9)
    ax.set_ylabel(None)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, imp_df["importance"].max() * 1.2)

    # 90% 처음 도달하는 인덱스 (0-based)
    cut_idx = np.argmax(imp_df["cumsum"].values >= threshold)

    # 주황색 rank 기준으로 +1
    cut_rank = (int(cut_idx) + 1) - 0.5

    # 90% 도달 지점 수직선 (핵심)
    plt.axhline(
        y=cut_rank,
        linestyle="--",
        color="red",
        alpha=0.8,
    )

    plt.tight_layout()
    plt.show()
    plt.close()

    return imp_df
