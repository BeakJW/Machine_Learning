from pandas import DataFrame, merge, concat, Series
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    log_loss,
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
    scoring=None,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
):
    """학습 곡선 분석 및 과적합 판정 함수 (회귀/분류 자동 판별)"""

    # 문제 유형 자동 판별
    is_classification = (
        hasattr(estimator, "_estimator_type")
        and estimator._estimator_type == "classifier"
    )

    # scoring 자동 설정
    if scoring is None:
        scoring = "roc_auc" if is_classification else "neg_root_mean_squared_error"

    # learning curve 계산
    train_sizes, train_scores, cv_scores = learning_curve(
        estimator=estimator,
        X=x,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        shuffle=False,
        random_state=52,
    )

    # 모델명 추출
    if hasattr(estimator, "named_steps"):
        classname = estimator.named_steps["model"].__class__.__name__
    else:
        classname = estimator.__class__.__name__

    # 회귀 전용 처리
    if not is_classification:
        # neg RMSE → RMSE
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

        # 마지막 두 지점 기울기
        train_slope = train_mean[-1] - train_mean[-2]
        cv_slope = cv_mean[-1] - cv_mean[-2]

        # 판정 로직
        if gap_ratio >= 0.95 and final_cv > some_threshold:
            status = "⚠️ 과소적합"
        elif gap_ratio <= 0.8 and train_slope > 0 and cv_slope < 0:
            status = "⚠️ 데이터 추가시 일반화 기대"
        elif gap_ratio <= 0.8:
            status = "⚠️ 과대적합"
        elif gap_ratio <= 0.95 and var_ratio <= 0.10:
            status = "✅ 일반화 양호"
        elif var_ratio > 0.15:
            status = "⚠️ 데이터 부족"
        else:
            status = "⚠️ 판단유보"

        metric_name = "RMSE"

    # 분류 전용 처리
    else:
        train_metric = train_scores
        cv_metric = cv_scores

        train_mean = train_metric.mean(axis=1)
        cv_mean = cv_metric.mean(axis=1)
        cv_std = cv_metric.std(axis=1)

        final_train = train_mean[-1]
        final_cv = cv_mean[-1]
        final_std = cv_std[-1]

        # 분류용 비율 정의 (차이 기반)
        gap_ratio = final_train - final_cv
        var_ratio = final_std

        # 분류 판정 로직
        if final_train < 0.6 and final_cv < 0.6:
            status = "⚠️ 과소적합"
        elif gap_ratio > 0.1:
            status = "⚠️ 과대적합"
        elif gap_ratio <= 0.05 and var_ratio <= 0.05:
            status = "✅ 일반화 양호"
        elif var_ratio > 0.1:
            status = "⚠️ 데이터 부족"
        else:
            status = "⚠️ 판단유보"

        metric_name = scoring.upper()

    # 정량 결과 표
    result_df = DataFrame(
        {
            f"Train {metric_name}": [final_train],
            f"CV {metric_name} 평균": [final_cv],
            f"CV {metric_name} 표준편차": [final_std],
            f"Train/CV 비율": [gap_ratio],
            f"CV 변동성 비율": [var_ratio],
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
        label=f"Train {metric_name}",
        ax=ax,
    )

    sb.lineplot(
        x=train_sizes,
        y=cv_mean,
        marker="o",
        markeredgecolor="#ffffff",
        label=f"CV {metric_name}",
        ax=ax,
    )

    ax.set_xlabel("훈련 데이터 수", fontsize=8, labelpad=5)
    ax.set_ylabel(metric_name, fontsize=8, labelpad=5)
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
    ax.set_xlabel("Permutation Importance (mean)", fontsize=10)
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
        "stable",
        "variable",
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
    """변수 중요도 계산 및 시각화 (XGBoost/일반 모델 자동 판별)"""
    try:
        from xgboost import XGBRegressor

        if isinstance(model, XGBRegressor):
            booster = model.get_booster()
            imp = booster.get_score(importance_type="gain")
            imp_sr = Series(imp)
            imp_df = DataFrame(imp_sr, columns=["importance"])
        else:
            raise ImportError  # 일반 모델 처리로 이동
    except (ImportError, AttributeError):
        perm = permutation_importance(
            estimator=model,
            X=x_train,
            y=y_train,
            scoring="r2",
            n_repeats=30,
            random_state=42,
            n_jobs=-1,
        )

        imp_df = DataFrame({"importance": perm.importances_mean}, index=x_train.columns)

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

    sb.barplot(data=df, x="importance", y=df.index, ax=ax)

    # 값 라벨 추가
    for i, v in enumerate(imp_df["importance"]):
        ax.text(
            v + 0.005,
            i,
            f"{v:.1f} ({imp_df.iloc[i]['cumsum']*100:.1f}%)",
            va="center",
        )

    ax.set_title("Feature Importance", fontsize=9)
    ax.set_xlabel("importance(cumsum)", fontsize=9)
    ax.set_ylabel(None)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, imp_df["importance"].max() * 1.2)

    # 90% 처음 도달하는 인덱스
    cut_idx = np.argmax(imp_df["cumsum"].values >= threshold)
    cut_rank = (int(cut_idx) + 1) - 0.5

    # 90% 도달 지점 수직선
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


def hs_describe(data, columns=None):
    """기술통계량 + 이상치 + 분포 분석"""
    num_columns = list(data.select_dtypes(include=np.number).columns)

    if not columns:
        columns = num_columns

    # 기술통계량 구하기
    desc = data[columns].describe().T

    # 각 컬럼별 결측치 수(na_count) 추가
    na_counts = data[columns].isnull().sum()
    desc.insert(1, "na_count", na_counts)

    # 결측치 비율(na_rate) 추가
    desc.insert(2, "na_rate", (na_counts / len(data)) * 100)

    # 추가 통계량 계산
    additional_stats = []
    for f in columns:
        # 숫자 타입이 아니면 건너뜀
        if f not in num_columns:
            continue

        # 사분위수
        q1 = data[f].quantile(q=0.25)
        q3 = data[f].quantile(q=0.75)

        # 이상치 경계 (Tukey's fences)
        iqr = q3 - q1
        down = q1 - 1.5 * iqr
        up = q3 + 1.5 * iqr

        # 왜도
        skew = data[f].skew()

        # 이상치 개수 및 비율
        outlier_count = ((data[f] < down) | (data[f] > up)).sum()
        outlier_rate = (outlier_count / len(data)) * 100

        # 분포 특성 판정 (왜도 기준)
        abs_skew = abs(skew)
        if abs_skew < 0.5:
            dist = "거의 대칭"
        elif abs_skew < 1.0:
            if skew > 0:
                dist = "약간 우측 꼬리"
            else:
                dist = "약간 좌측 꼬리"
        elif abs_skew < 2.0:
            if skew > 0:
                dist = "중간 우측 꼬리"
            else:
                dist = "중간 좌측 꼬리"
        else:
            if skew > 0:
                dist = "극단 우측 꼬리"
            else:
                dist = "극단 좌측 꼬리"

        # 로그변환 필요성 판정
        if abs_skew < 0.5:
            log_need = "낮음"
        elif abs_skew < 1.0:
            log_need = "중간"
        else:
            log_need = "높음"

        additional_stats.append(
            {
                "field": f,
                "iqr": iqr,
                "up": up,
                "down": down,
                "outlier_count": outlier_count,
                "outlier_rate": outlier_rate,
                "skew": skew,
                "dist": dist,
                "log_need": log_need,
            }
        )

    additional_df = DataFrame(additional_stats).set_index("field")

    # 결과 병합
    result = concat([desc, additional_df], axis=1)

    return result


def category_describe(data, columns=None):
    """범주형 변수 분석 (빈도표 + 요약)"""
    num_columns = data.select_dtypes(include=np.number).columns

    if not columns:
        columns = data.select_dtypes(include=["object", "category", "bool"]).columns

    result = []
    summary = []
    for f in columns:
        # 숫자형 컬럼은 건너뜀
        if f in num_columns:
            continue

        # 각 범주의 빈도수 계산 (NaN 포함)
        value_counts = data[f].value_counts(dropna=False)

        # 범주별 빈도/비율 정보 추가
        for category, count in value_counts.items():
            rate = (count / len(data)) * 100
            result.append(
                {"변수": f, "범주": category, "빈도": count, "비율(%)": round(rate, 2)}
            )

        if len(value_counts) == 0:
            continue

        # 최다/최소 범주 정보 추가
        max_category = value_counts.index[0]
        max_count = value_counts.iloc[0]
        max_rate = (max_count / len(data)) * 100
        min_category = value_counts.index[-1]
        min_count = value_counts.iloc[-1]
        min_rate = (min_count / len(data)) * 100
        summary.append(
            {
                "변수": f,
                "최다_범주": max_category,
                "최다_비율(%)": round(max_rate, 2),
                "최소_범주": min_category,
                "최소_비율(%)": round(min_rate, 2),
            }
        )

    return DataFrame(result), DataFrame(summary).set_index("변수")


def hs_cls_bin_scores(estimator, x_test, y_test):
    """이진 분류 모델 성능 평가 및 ROC 곡선"""
    # 예측 확률
    y_pred_proba = estimator.predict_proba(x_test)
    y_pred_proba_1 = estimator.predict_proba(x_test)[:, 1]

    # 예측값
    y_pred = estimator.predict(x_test)

    # 의사결정계수
    log_loss_test = -log_loss(y_test, y_pred_proba, normalize=False)
    y_null = np.ones_like(y_test) * y_test.mean()
    log_loss_null = -log_loss(y_test, y_null, normalize=False)
    pseudo_r2 = 1 - (log_loss_test / log_loss_null)

    # 혼동행렬
    cm = confusion_matrix(y_test, y_pred)
    ((TN, FP), (FN, TP)) = cm

    # 클래스 이름
    if hasattr(estimator, "named_steps"):
        classname = estimator.named_steps["model"].__class__.__name__
    else:
        classname = estimator.__class__.__name__

    # auc score
    auc = roc_auc_score(y_test, y_pred_proba_1)

    score_df = DataFrame(
        {
            "정확도(Accuracy)": [accuracy_score(y_test, y_pred)],
            "정밀도(Precision)": [precision_score(y_test, y_pred)],
            "재현율(Recall,tpr)": [recall_score(y_test, y_pred)],
            "위양성율(Fallout,fpr)": [FP / (TN + FP)],
            "특이성(TNR)": [1 - (FP / (TN + FP))],
            "F1 Score": [f1_score(y_test, y_pred)],
            "AUC": [auc],
        },
        index=[classname],
    )

    # ROC 곡선 그리기
    roc_fpr, roc_tpr, thresholds = roc_curve(y_test, y_pred_proba_1)

    width_px = 1000
    height_px = 900
    rows = 1
    cols = 1
    figsize = (width_px / my_dpi, height_px / my_dpi)
    fig, ax = plt.subplots(rows, cols, figsize=figsize, dpi=my_dpi)

    sb.lineplot(x=roc_fpr, y=roc_tpr, ax=ax)
    sb.lineplot(x=[0, 1], y=[0, 1], color="red", linestyle=":", alpha=0.5, ax=ax)
    plt.fill_between(x=roc_fpr, y1=roc_tpr, alpha=0.1)

    ax.grid(True, alpha=0.3)
    ax.set_title(f"AUC={auc:.4f}", fontsize=10, pad=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
    plt.close()

    return score_df
