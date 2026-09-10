"""
==============================================================================
AI-Driven Anomaly Detection and Predictive Maintenance in Cloud DevOps
Phase 13: Final Results Consolidation
==============================================================================

Consolidates all verified experimental metrics across:
  1. Alibaba Dense Autoencoder Baseline
  2. Bitbrains Dense Autoencoder (Fair Baseline)
  3. Bitbrains Seq2Seq LSTM-Autoencoder
  4. Controlled Synthetic Anomaly Benchmark
  5. Validation-Derived Feature Explanations

Strict Constraints:
  - Read-Only access to existing models, scalers, and sequences.
  - Zero retraining or modification of existing outputs.
  - Exact preservation of verified numbers; unrecorded values set to 'N/A'.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_final_model_comparison_csv():
    """Builds final_model_comparison.csv with verified metrics from source outputs."""
    records = [
        {
            "Model": "Alibaba Dense Autoencoder",
            "Dataset": "Alibaba Cluster Data",
            "Architecture": "Dense(4-2-4-5)",
            "InputShape": "(5,)",
            "Parameters": 71,
            "TrainingTimeSeconds": "N/A",
            "ValidationMSE": "N/A",
            "TestMeanMSE": 0.202358,
            "ValidationThreshold": "N/A",
            "TestAnomalyCount": 23,
            "TestAnomalyRate": "5.10%",
            "SyntheticAccuracy": 0.953229,
            "SyntheticPrecision": 0.661017,
            "SyntheticRecall": 0.975000,
            "SyntheticF1": 0.787879,
            "SyntheticInferenceTimeSeconds": "N/A",
        },
        {
            "Model": "Bitbrains Dense Autoencoder",
            "Dataset": "Bitbrains Cloud Telemetry",
            "Architecture": "Dense(32-16-8-16-32-60)",
            "InputShape": "(60,)",
            "Parameters": 5284,
            "TrainingTimeSeconds": 83.66,
            "ValidationMSE": 0.119368,
            "TestMeanMSE": 0.127509,
            "ValidationThreshold": 0.359623,
            "TestAnomalyCount": 797,
            "TestAnomalyRate": "4.51%",
            "SyntheticAccuracy": 0.935000,
            "SyntheticPrecision": 1.000000,
            "SyntheticRecall": 0.870000,
            "SyntheticF1": 0.930481,
            "SyntheticInferenceTimeSeconds": 0.34,
        },
        {
            "Model": "Bitbrains LSTM Autoencoder",
            "Dataset": "Bitbrains Cloud Telemetry",
            "Architecture": "Seq2Seq LSTM(64-32-32-64-5)",
            "InputShape": "(12, 5)",
            "Parameters": 16549,
            "TrainingTimeSeconds": 813.61,
            "ValidationMSE": 0.078984,
            "TestMeanMSE": 0.095711,
            "ValidationThreshold": 0.277209,
            "TestAnomalyCount": 1221,
            "TestAnomalyRate": "6.91%",
            "SyntheticAccuracy": 0.993000,
            "SyntheticPrecision": 1.000000,
            "SyntheticRecall": 0.986000,
            "SyntheticF1": 0.992951,
            "SyntheticInferenceTimeSeconds": 1.54,
        },
    ]

    df = pd.DataFrame(records)
    out_path = OUTPUT_DIR / "final_model_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Created: {out_path} ({len(df)} rows)")
    return df


def generate_final_explanation_summary_csv():
    """Builds final_explanation_summary.csv from Phase 11C explanation outputs."""
    records = [
        {
            "Feature": "net_out",
            "TopContributorCount": 767,
            "TopContributorSharePercent": 62.82,
            "AverageContributionPercent": 47.71,
        },
        {
            "Feature": "disk_io",
            "TopContributorCount": 251,
            "TopContributorSharePercent": 20.56,
            "AverageContributionPercent": 20.24,
        },
        {
            "Feature": "net_in",
            "TopContributorCount": 106,
            "TopContributorSharePercent": 8.68,
            "AverageContributionPercent": 14.27,
        },
        {
            "Feature": "mem",
            "TopContributorCount": 80,
            "TopContributorSharePercent": 6.55,
            "AverageContributionPercent": 12.11,
        },
        {
            "Feature": "cpu",
            "TopContributorCount": 17,
            "TopContributorSharePercent": 1.39,
            "AverageContributionPercent": 5.68,
        },
    ]

    df = pd.DataFrame(records)
    out_path = OUTPUT_DIR / "final_explanation_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Created: {out_path} ({len(df)} rows)")
    return df


def generate_final_results_plot():
    """Creates a 4-panel visual comparison between Bitbrains Dense AE and LSTM-AE."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2))
    fig.patch.set_facecolor("#FAFAFC")

    models = ["Dense AE", "LSTM-AE"]
    colors = ["#2B6CB0", "#C53030"]

    # Panel 1: Validation MSE (Lower is Better)
    val_mse = [0.119368, 0.078984]
    bars1 = axes[0].bar(models, val_mse, color=colors, width=0.52, edgecolor="black", linewidth=1.1)
    axes[0].set_title("Validation MSE\n(Lower is Better)", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_ylabel("Mean Squared Error (MSE)", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 0.16)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].text(0.5, 0.88, "-33.8% Loss", transform=axes[0].transAxes, ha="center", color="#C53030", fontweight="bold", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF5F5", edgecolor="#FEB2B2"))

    # Panel 2: Test Mean Reconstruction Error (Lower is Better)
    test_mse = [0.127509, 0.095711]
    bars2 = axes[1].bar(models, test_mse, color=colors, width=0.52, edgecolor="black", linewidth=1.1)
    axes[1].set_title("Test Mean Reconstruction Error\n(Lower is Better)", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_ylabel("Mean Squared Error (MSE)", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 0.17)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[1].text(0.5, 0.88, "-24.9% Loss", transform=axes[1].transAxes, ha="center", color="#C53030", fontweight="bold", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF5F5", edgecolor="#FEB2B2"))

    # Panel 3: Synthetic Recall (Higher is Better)
    rec = [87.00, 98.60]
    bars3 = axes[2].bar(models, rec, color=colors, width=0.52, edgecolor="black", linewidth=1.1)
    axes[2].set_title("Synthetic Anomaly Recall\n(Higher is Better)", fontsize=12, fontweight="bold", pad=10)
    axes[2].set_ylabel("Detection Rate / Recall (%)", fontsize=10, fontweight="bold")
    axes[2].set_ylim(70, 108)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars3:
        h = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[2].text(0.5, 0.88, "+11.6% Recall", transform=axes[2].transAxes, ha="center", color="#C53030", fontweight="bold", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF5F5", edgecolor="#FEB2B2"))

    # Panel 4: Synthetic F1-Score (Higher is Better)
    f1 = [93.05, 99.30]
    bars4 = axes[3].bar(models, f1, color=colors, width=0.52, edgecolor="black", linewidth=1.1)
    axes[3].set_title("Synthetic Benchmark F1-Score\n(Higher is Better)", fontsize=12, fontweight="bold", pad=10)
    axes[3].set_ylabel("F1-Score (%)", fontsize=10, fontweight="bold")
    axes[3].set_ylim(70, 108)
    axes[3].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars4:
        h = bar.get_height()
        axes[3].text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[3].text(0.5, 0.88, "+6.25% F1", transform=axes[3].transAxes, ha="center", color="#C53030", fontweight="bold", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF5F5", edgecolor="#FEB2B2"))

    for ax in axes:
        ax.tick_params(axis="x", labelsize=11)
        ax.set_facecolor("#FFFFFF")

    fig.suptitle("Bitbrains Experimental Comparison: Dense Autoencoder (60D) vs. Seq2Seq LSTM-Autoencoder", 
                 fontsize=15, fontweight="bold", y=0.98)

    plt.figtext(0.5, 0.01, 
                "Controlled synthetic anomaly benchmark on N = 4,000 cases (2,000 normal + 2,000 contiguous synthetic anomalies). "
                "Fixed validation thresholds (Dense: 0.3596, LSTM: 0.2772). Does not represent real-world failure prediction.",
                ha="center", fontsize=10, style="italic", color="#4A5568")

    out_path = OUTPUT_DIR / "final_results_comparison.png"
    plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.30)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Created: {out_path}")


def generate_final_results_summary_md():
    """Generates the comprehensive research report markdown."""
    report_content = """# Final Experimental Results

## 1. Research Setup
This project investigated unsupervised reconstruction-based anomaly detection and feature attribution for cloud DevOps environments across two public cloud workload datasets:

1. **Alibaba Cluster Trace Data**: Used during initial project phases (Phases 1–5) to establish proof-of-concept autoencoder modeling, error thresholding, synthetic spike testing, and sequential degradation analysis on isolated server-level metrics.
2. **Bitbrains Cloud Telemetry**: Introduced in Phase 10 to evaluate real multi-tenant cloud virtual machine (VM) workloads across continuous temporal sequences. Telemetry channels include compute (`cpu_util_percent`), memory (`mem_util_percent`), network throughput (`net_in`, `net_out`), and disk activity (`disk_io_percent`). Ten representative VMs spanning typical, high-disk, high-network, high-memory, and high-CPU usage profiles were selected.
3. **Fair Model Baseline Alignment**: To evaluate whether recurrent temporal architectures offer genuine advantages over feedforward baselines, a **Dense Autoencoder** was trained on the exact same 12-step temporal sequences flattened into 60-dimensional vectors ($12 \\times 5$). Both models were evaluated under identical, strictly validation-derived anomaly thresholds.

---

## 2. Alibaba Baseline
The Alibaba dataset established initial baseline feasibility using an early feedforward autoencoder:
- **Architecture**: `Input(5) -> Dense(4, relu) -> Dense(2, relu) -> Dense(4, relu) -> Dense(5, linear)` (71 trainable parameters).
- **Test Reconstruction Error**: Mean MSE of `0.202358` across 451 test observations.
- **Initial Anomaly Thresholding**: Using an empirical 95th percentile threshold (`0.603711`) derived from the test set error distribution, `23` out of `451` test observations (`5.10%`) were flagged as anomalous.
- **Synthetic Evaluation**: On 40 injected synthetic single-step anomalies mixed into 411 normal points ($N = 451$), the Alibaba baseline achieved:
  - Accuracy: `0.9532`
  - Precision: `0.6610`
  - Recall: `0.9750` (39/40 detected)
  - F1-Score: `0.7879`
  - Confusion Matrix: TP = 39, FP = 20, FN = 1, TN = 389.
- **Degradation Trend Analysis**: Demonstrative rolling window calculations showed sequential trends in reconstruction residuals, but did not incorporate progressive degradation mechanics.

---

## 3. Bitbrains Dense vs. LSTM Reconstruction Performance

| Metric / Dimension | Bitbrains Dense Autoencoder | Bitbrains LSTM-Autoencoder | Relative Difference |
| :--- | :---: | :---: | :---: |
| **Model Architecture** | `Dense(32-16-8-16-32-60)` | `Seq2Seq LSTM(64-32-32-64-5)` | Feedforward vs. Recurrent |
| **Input Representation** | Flattened vector: `(60,)` | Temporal sequence: `(12, 5)` | Static vs. Sequence |
| **Trainable Parameters** | **5,284** | **16,549** | Dense has $68.1\%$ fewer params |
| **Training Duration** | **83.66 seconds** (30 epochs) | **813.61 seconds** (30 epochs) | Dense trains $\sim 9.7\\times$ faster |
| **Validation Loss (Best MSE)** | `0.119368` | `0.078984` | **LSTM achieves 33.8% lower val loss** |
| **Validation Threshold (95th %ile)** | `0.359623` | `0.277209` | LSTM has tighter normal bounds |
| **Test Mean Reconstruction MSE** | `0.127509` | `0.095711` | **LSTM achieves 24.9% lower test error** |
| **Test Sequences Evaluated** | 17,680 sequences | 17,680 sequences | Identical test set |
| **Test Anomalies Detected** | 797 sequences (`4.51%`) | 1,221 sequences (`6.91%`) | LSTM detects $+424$ subtle anomalies |

---

## 4. Controlled Synthetic Anomaly Evaluation

A controlled benchmark was executed on exactly **4,000 identical cases** ($50.0\\%$ anomaly prevalence) evaluated by both models using fixed validation-derived operational thresholds:
- **2,000 Clean Controls**: Unmodified normal sequences ($MSE < \\tau_{\\text{val}}$ in both models).
- **2,000 Contiguous Synthetic Spikes**: 400 cases per telemetry channel (`cpu`, `mem`, `net_in`, `net_out`, `disk_io`) injected with $+2.5\\sigma$ to $+4.5\\sigma$ additive shifts over 3–6 contiguous timesteps.

### Overall Benchmark Metrics ($N = 4,000$)

| Model Architecture | Accuracy | Precision | Recall (Sensitivity) | F1-Score | Specificity | True Positives | False Positives | False Negatives | True Negatives | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Autoencoder** | **0.9350** | **1.0000** | **0.8700** | **0.9305** | **1.0000** | 1,740 | 0 | 260 | 2,000 | **0.34s** |
| **LSTM Autoencoder** | **0.9930** | **1.0000** | **0.9860** | **0.9930** | **1.0000** | 1,972 | 0 | 28 | 2,000 | **1.54s** |

### Detection Rate (Recall) Breakdown by Channel (400 cases each)

| Telemetry Channel | Dense AE Detection Rate (%) | LSTM-AE Detection Rate (%) | Dense Mean MSE | LSTM Mean MSE | Sensitivity Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CPU Spike (`cpu_util_percent`)** | 92.75% (371/400) | **98.50%** (394/400) | 0.6740 | 0.7999 | +5.75% (LSTM) |
| **Memory Spike (`mem_util_percent`)** | 88.50% (354/400) | **98.25%** (393/400) | 0.6873 | 0.8851 | +9.75% (LSTM) |
| **Net-In Spike (`net_in`)** | 91.00% (364/400) | **100.00%** (400/400) | 0.6879 | 1.0058 | +9.00% (LSTM) |
| **Net-Out Spike (`net_out`)** | 71.00% (284/400) | **97.50%** (390/400) | 0.5628 | 0.8359 | **+26.50%** (LSTM) |
| **Disk I/O Spike (`disk_io_percent`)** | 91.75% (367/400) | **98.75%** (395/400) | 0.6616 | 0.9214 | +7.00% (LSTM) |

> **This is a controlled synthetic anomaly evaluation and does not represent real-world production failure prediction.**

---

## 5. Anomaly Explanation
Feature-level squared reconstruction error attribution was computed across all 12 timesteps for the **1,221 test sequences** flagged by the LSTM-AE under the final validation threshold:

$$\\text{MSE}_j = \\frac{1}{12} \\sum_{t=0}^{11} (x(t,j) - \\hat{x}(t,j))^2, \\quad \\text{Contribution}_j = \\frac{\\text{MSE}_j}{\\sum_{k=1}^5 \\text{MSE}_k} \\times 100\\%$$

### Telemetry Channel Attribution ($N_{\\text{anomalies}} = 1,221$)

| Telemetry Channel | Top Contributor Count | Top Contributor Share (%) | Average Contribution (%) | Primary Observable Pattern |
| :--- | :---: | :---: | :---: | :--- |
| **Network Out (`net_out`)** | **767** | **62.82%** | **47.71%** | Bursty egress transfers and sync jobs |
| **Disk I/O (`disk_io`)** | **251** | **20.56%** | **20.24%** | Periodic flush and commit bursts |
| **Network In (`net_in`)** | **106** | **8.68%** | **14.27%** | Ingress data ingestion bursts |
| **Memory (`mem`)** | **80** | **6.55%** | **12.11%** | Caching footprint fluctuations |
| **CPU (`cpu`)** | **17** | **1.39%** | **5.68%** | Scheduled batch compute spikes |
| **Total** | **1,221** | **100.00%** | **100.00%** | Comprehensive attribution |

> **Operational Guardrail**:
> Feature contributions are reconstruction-residual attribution and do not establish causal root cause or failure probability.

---

## 6. Key Findings

1. **Recurrent Modeling Reduces Reconstruction Residuals**: The Seq2Seq LSTM-AE achieved a $33.8\\%$ lower validation loss (`0.078984` vs. `0.119368`) and a $24.9\\%$ lower test reconstruction error (`0.095711` vs. `0.127509`) compared to the fair Dense AE baseline on identical 12-step windows.
2. **Superior Sensitivity to Multi-Step Workload Spikes**: In controlled synthetic testing across 4,000 benchmark cases, LSTM-AE achieved an overall recall of **$98.60\\%$** versus **$87.00\\%$** for Dense AE, reducing missed anomalies from 260 down to 28.
3. **Severe Dense Degradation on Bursty Channels**: The flattened Dense AE experienced a sharp performance drop on network egress (`net_out`), detecting only **$71.00\\%$** of synthetic spikes, whereas the LSTM-AE maintained **$97.50\\%$** detection by capturing sequential auto-correlation.
4. **Efficiency vs. Fidelity Trade-off**: The Dense AE trained $\\sim 9.7\\times$ faster ($83.66\\text{s}$ vs. $813.61\\text{s}$) with $68.1\\%$ fewer parameters ($5,284$ vs. $16,549$) and inferred $3.4\\times$ faster ($0.34\\text{s}$ vs. $1.54\\text{s}$ for 4,000 cases).
5. **Operational Anomaly Concentration**: In real Bitbrains test data, $83.38\\%$ of validation-flagged anomalies were driven by network egress ($62.82\\%$) and disk I/O ($20.56\\%$), reflecting the bursty I/O nature of virtualized enterprise workloads.

---

## 7. Limitations

1. **No Ground-Truth Production Labels**: The Bitbrains dataset contains raw, unlabelled telemetry traces. Unsupervised anomalies indicate deviation from learned baseline reconstruction, not confirmed operational outages or bugs.
2. **Synthetic Benchmarks are Controlled Approximations**: The synthetic evaluation used fixed additive shifts ($+2.5\\sigma$ to $+4.5\\sigma$) over contiguous windows. Real-world failures exhibit complex, heterogeneous fault modes (e.g. cascading memory leaks, deadlock stalls, thrashing).
3. **Selected VM Subsampling**: Experiments focused on 10 representative VMs ($71,032$ training sequences, $17,680$ test sequences) to preserve strict methodological rigor within computational constraints, rather than indexing all 1,241 available Bitbrains trace files.
4. **Empirical Operational Thresholding**: The 95th percentile validation error threshold is an empirical heuristic selected for controlled comparison. Practical production deployment requires tuning thresholds to organizational alert fatigue tolerances and SLA constraints.
5. **Absence of Progressive Failure Forecasting**: The sequential rolling degradation analysis illustrates telemetry drift but does not demonstrate predictive time-to-failure forecasting or causal degradation trajectories.

---

## 8. Final Conclusion

This investigation demonstrates that incorporating temporal sequence modeling via a Seq2Seq LSTM-Autoencoder provides measurable advantages in reconstruction fidelity and anomaly detection sensitivity over a feedforward Dense Autoencoder on virtualized cloud telemetry traces. The LSTM architecture reduces reconstruction loss by $\\sim 25\\text{--}34\\%$ and significantly outperforms flattened dense representations on bursty network channels ($97.5\\%$ vs. $71.0\\%$ recall).

However, these experimental findings do not guarantee that LSTM architectures will universally dominate in production environments. Dense autoencoders offer nearly an order of magnitude faster training and $3.4\\times$ higher inference throughput with a $68\\%$ smaller memory footprint, making them highly attractive for resource-constrained edge metric collection. An optimal production DevOps architecture should balance these trade-offs: leveraging lightweight feedforward autoencoders for high-throughput edge filtering and recurrent sequence autoencoders for mission-critical core service telemetry.
"""

    out_path = OUTPUT_DIR / "final_results_summary.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[OK] Created: {out_path}")


def main():
    print("=" * 70)
    print("PHASE 13: FINAL RESULTS CONSOLIDATION")
    print("=" * 70)

    # 1. Comparison CSV
    df_comparison = generate_final_model_comparison_csv()

    # 2. Explanation CSV
    df_explanation = generate_final_explanation_summary_csv()

    # 3. Visualization PNG
    generate_final_results_plot()

    # 4. Comprehensive Markdown Report
    generate_final_results_summary_md()

    print("\n" + "=" * 70)
    print("ALL PHASE 13 DELIVERABLES GENERATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
