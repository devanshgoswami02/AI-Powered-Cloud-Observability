# Bitbrains Dataset Compatibility & Telemetry Analysis Report

**Investigation Scope**: Standalone inspection of the **Bitbrains GWA-T-12 `fastStorage`** dataset (`2013-8` release) located at `C:\Users\goswa\OneDrive\Desktop\PROJECTS\DOWNLOADS\gwa_t_12_fastStorage\fastStorage\2013-8`.  
**Baseline Integrity**: The Alibaba-trained Dense Autoencoder baseline, its training pipeline, scripts, output CSVs, and model weights remain **frozen and unmodified**. No models were trained, no existing scalers applied, and no data was copied into the project directory.

---

## 1. Executive Summary

This report establishes the structural, statistical, and operational compatibility between the **Bitbrains GWA-T-12 `fastStorage`** virtual machine telemetry dataset and the project's baseline **Alibaba Cloud DevOps Anomaly Detection** pipeline.

The baseline model operates on an 8-day Alibaba cluster dataset aggregated into 300-second intervals across five standardized resource metrics (`cpu_util_percent`, `mem_util_percent`, `net_in`, `net_out`, `disk_io_percent`). This study determines whether and how Bitbrains can be utilized in Phase 9A research as an independent validation set, a separate training/validation benchmark, or a cross-domain generalization experiment.

```
================================================================================
BITBRAINS DATASET COMPATIBILITY SUMMARY (GWA-T-12 fastStorage 2013-8)
================================================================================
Files Analyzed               : 1,250 VM CSV traces (1.19 GB total)
Total Fleet Observations     : 11,221,800 rows
Observations Per VM          : Min: 5 | Median: 8,618 | Mean: 8,977 | Max: 23,692
Full 30-Day Traces (>=8000)  : 1,154 VMs (92.32%)
Short Traces (<100)          : 9 VMs (0.72%)
Schema Consistency           : 100% across all 1,250 files (11 columns, delimiter: ';\t')
Missing / Blank Cells        : 0 (clean table structure)

Temporal Properties:
- Raw Nominal Interval       : 300 seconds (5 minutes), NOT 300 ms
- Exact 300s Cadence         : 83.61% of all row transitions
- Gaps / Irregular Intervals : 16.39% (Max gap: 2,100s / 35 min)

Resource Metric Characteristics:
- CPU Usage [%]              : Mean: 8.00% | Median: 0.87% | Zero Rate: 31.34% | Max: 112.5%
- Memory Capacity == 0       : 15.36% of rows (causes division-by-zero in raw ratio)
- Memory Utilization [%]     : Mean: 9.41% | Median: 4.28% | Outliers >100%: 0.001%
- Network Throughput (KB/s)  : Mean: 63.2 (In) / 58.3 (Out) | Zero Rate: ~50-55% | Fat tail
- Total Disk Throughput (KB/s): Mean: 300.5 | Median: 1.0 | Read Zero Rate: 91.88% | Max: 618 MB/s

Five-Feature Mapping         : Structurally practical; requires unit handling
Technical Feasibility (300s) : Fully feasible (native cadence is 300 seconds)
Recommended Research Role    : Separate Training/Validation Dataset & Domain-Shift Benchmark
================================================================================
```

---

## 2. Facts Directly Measured from Bitbrains

### 2.1 Dataset Size & Volume
* **Total VM CSV Files**: `1,250` individual virtual machine telemetry files.
* **Total Size on Disk**: `1,188.83 MB` (~1.19 GB).
* **Average File Size**: `973.89 KB` (Min: `819 B`, Max: `1.76 MB`).
* **Total Observations Across Fleet**: `11,221,800` rows.

### 2.2 Observations Per VM
* **Minimum Observations**: `5`
* **Maximum Observations**: `23,692`
* **Median Observations**: `8,618.0`
* **Mean Observations**: `8,977.44`
* **Standard Deviation**: `3,045.98`
* **Full-Trace Prevalence**: `1,154` out of 1,250 VMs (**92.32%**) contain $\ge 8,000$ observations (approaching a full 30-day monitoring window: $30 \text{ days} \times 24 \text{ hours} \times 12 \text{ intervals/hour} = 8,640$ nominal 5-minute intervals). The most common trace lengths are `8,616`, `8,614`, `8,615`, `8,617`, and `8,635`.

### 2.3 Short Trace Analysis
* **Traces with $< 10$ observations**: `3` VMs
* **Traces with $< 100$ observations**: `9` VMs
* **Traces with $< 500$ observations**: `12` VMs
* **Traces with $< 1,000$ observations**: `16` VMs  
*(These short traces reflect short-lived, transient, or early-terminated VMs).*

### 2.4 Schema & Column Consistency
* **Consistency**: **100% identical** across all 1,250 files (0 schema mismatches).
* **Delimiter**: Semicolon with tab formatting (`;\t`).
* **Exact Raw Columns (11 total)**:
  1. `Timestamp [ms]`
  2. `CPU cores`
  3. `CPU capacity provisioned [MHZ]`
  4. `CPU usage [MHZ]`
  5. `CPU usage [%]`
  6. `Memory capacity provisioned [KB]`
  7. `Memory usage [KB]`
  8. `Disk read throughput [KB/s]`
  9. `Disk write throughput [KB/s]`
  10. `Network received throughput [KB/s]`
  11. `Network transmitted throughput [KB/s]`

### 2.5 Missing Values & Numeric Parsing
* **Missing/Blank Cells**: `0` across the dataset.
* **Negative Values**: `0` across all metric columns.
* **Numeric Parsing**: All fields parse as floating-point numbers without syntax errors. Memory values frequently utilize standard scientific notation (e.g., `6.7108864E7`), which standard parsers interpret seamlessly.

### 2.6 Timestamp Interval Statistics
* **Column Header vs Actual Value**: The column header states `Timestamp [ms]`, but values are standard Unix epoch timestamps recorded in **seconds** (e.g., `1376314846` corresponding to August 12, 2013).
* **Time Deltas Between Successive Rows**:
  * **Median Interval**: `300.0 s` (5 minutes)
  * **Mean Interval**: `275.07 s`
  * **Exact 300s Sampling Frequency**: **`83.61%`** of all intervals are exactly 300 seconds (`936,272` out of `1,119,833` sample deltas).
  * **Intervals $< 300\text{ s}$**: `150,105` (includes timestamp duplicates $\Delta t = 0\text{ s}$ and short burst intervals).
  * **Intervals $> 300\text{ s}$ (Gaps)**: `33,456` (occasional monitoring dropouts or VM suspensions, with a maximum gap of `2,100 s` / 35 minutes).

### 2.7 Measured Resource Metric Distributions (Empirical Sample)

| Metric | Min | 25% | Median | Mean | 75% | 95% | 99% | Max | Std Dev | Zero / Idle Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU Usage [%]** | 0.00% | 0.00% | 0.87% | **8.00%** | 2.00% | 74.10% | 103.69% | **112.50%** | 22.57% | **31.34%** |
| **Memory Capacity [KB]** | 0.00 | 2.10E6 | 4.19E6 | 8.21E6 | 8.39E6 | 3.36E7 | 6.71E7 | 1.34E8 | 1.38E7 | **15.36%** |
| **Memory Usage [KB]** | 0.00 | 2.30E4 | 2.05E5 | 1.04E6 | 9.87E5 | 5.21E6 | 1.48E7 | 3.82E7 | 2.45E6 | **15.37%** |
| **Net Received [KB/s]** | 0.00 | 0.00 | 0.00 | **63.22** | 3.67 | 52.60 | 710.75 | 51,986.87 | 859.80 | **55.76%** |
| **Net Transmitted [KB/s]**| 0.00 | 0.00 | 0.07 | **58.28** | 1.00 | 24.13 | 303.80 | 41,243.73 | 866.73 | **49.79%** |
| **Disk Read [KB/s]** | 0.00 | 0.00 | 0.00 | **204.62** | 0.00 | 2.67 | 6,519.95 | 618,253.73 | 3,092.55 | **91.88%** |
| **Disk Write [KB/s]** | 0.00 | 0.00 | 1.00 | **95.85** | 8.27 | 49.07 | 750.03 | 56,630.60 | 998.76 | **28.16%** |

---

## 3. Transformations We Define

### 3.1 Five-Feature Mathematical Mapping
To align Bitbrains with the project's 5-feature baseline space:
1. `cpu_util_percent` $\leftarrow$ `CPU usage [%]`
2. `mem_util_percent` $\leftarrow$ $\frac{\text{Memory usage [KB]}}{\text{Memory capacity provisioned [KB]}} \times 100$
3. `net_in` $\leftarrow$ `Network received throughput [KB/s]`
4. `net_out` $\leftarrow$ `Network transmitted throughput [KB/s]`
5. `disk_io` $\leftarrow$ `Disk read throughput [KB/s]` $+$ `Disk write throughput [KB/s]`

### 3.2 Statistics of the Derived Features
* **Calculated Memory Utilization [%]**:
  * Measured values: Min: `0.00%`, Median: `4.28%`, Mean: `9.41%`, 75th: `10.99%`, 95th: `39.60%`, 99th: `55.60%`.
  * Outliers $> 100\%$: Observed in rare instances (max `854.35%`, 15 occurrences in 1.1M rows) caused by dynamic memory ballooning / unballooning where provisioned capacity was reported lower than instantaneous guest memory usage.
* **Derived Total Disk Throughput [KB/s]**:
  * Median: `1.00 KB/s`, Mean: `300.47 KB/s`, 95th: `94.67 KB/s`, 99th: `14,276.13 KB/s`, Max: `618,271.87 KB/s` (~618 MB/s), Zero rate: `28.14%`.

---

## 4. Empirical Data-Quality Findings

1. **Header Unit Discrepancy**: The column is named `Timestamp [ms]`, but values are Unix epoch seconds.
2. **Memory Capacity Division-by-Zero**: Exactly **15.36%** of rows have `Memory capacity provisioned [KB] == 0` (and `Memory usage [KB] == 0`). In these rows, computing $\frac{\text{Memory usage}}{\text{Memory capacity}} \times 100$ results in `NaN` ($0 / 0$) or `inf`.
3. **CPU Bursting Beyond 100%**: In **3.08%** of rows, `CPU usage [%]` exceeds 100% (up to 112.5%). This is an inherent property of ESXi hypervisor turbo/burst frequencies where utilized MHz exceeds baseline provisioned MHz.
4. **Extreme Positive Skew & Zero Inflation**:
   - Disk read throughput is idle (**91.88% zero**).
   - Network in is idle (**55.76% zero**).
   - Mean disk read (204.6 KB/s) is driven by extreme tail bursts up to 618 MB/s (standard deviation is $15\times$ the mean).

---

## 5. Architectural Comparison: Alibaba Baseline vs. Bitbrains

| Metric Dimension | Alibaba Baseline Dataset (Frozen) | Bitbrains `fastStorage` (GWA-T-12) | Direct Transfer Viability |
| :--- | :--- | :--- | :--- |
| **System Scope** | Aggregated cluster workload (2,243 rows) | 1,250 individual guest VMs (11.2M rows) | Structural mismatch (fleet aggregated vs single VM). |
| **Nature of Telemetry** | Production cloud host machine cluster | Real enterprise cloud VM workload traces | Both are real DevOps infrastructure metrics. |
| **Sampling Period** | 300-second grouped intervals | 300-second raw sampling (83.6% exact) | **Compatible time resolution** (5 minutes). |
| **CPU Utilization** | Mean: `40.18%`, range: [14.4%, 79.1%] | Mean: `8.00%`, median: `0.87%`, 31% zero | Massive distribution shift (busy cluster vs idle VMs). |
| **Memory Utilization**| Mean: `87.97%`, range: [77.0%, 95.0%] | Mean: `9.41%`, median: `4.28%`, 15% missing | Severe distribution shift + division-by-zero risk. |
| **Network Metrics** | Normalized indices ~`41` and ~`32.5` | Physical throughput (KB/s), 0 to 52,000 KB/s | **Unit mismatch** (Index vs KB/s). |
| **Disk I/O** | `disk_io_percent` [1.8%, 25.8%] | Throughput in KB/s, up to 618,000 KB/s | **Unit mismatch** (Percent vs Throughput). |

---

## 6. Technical Feasibility & Research Recommendations

### 6.1 Practicality of the Five-Feature Representation
* **Yes, conceptually and structurally**: The five core resources (CPU, Memory, Network In, Network Out, Disk I/O) exist and capture cloud resource dynamics.
* **However, direct zero-shot scoring is invalid**: Because the Alibaba baseline was trained on normalized indices and percentages ($Z$-score scaled on Alibaba means/stds), passing raw KB/s throughput values and zero-dominated VM metrics through the existing Alibaba scaler and model will cause catastrophic false positives.

### 6.2 Required Preprocessing for Bitbrains
1. **Division-by-Zero Handling**: Filter out inactive VM observations where `Memory capacity provisioned == 0`, or impute utilization as `0.0%` when usage is 0.
2. **Cap / Scale Memory Outliers**: Cap memory utilization at `100.0%` (or record ballooning as an explicit flag).
3. **Log-Transform Heavy Tails**: Apply $\log(1 + x)$ or quantile transformation to `net_in`, `net_out`, and `disk_io` to compress the extreme multi-magnitude range (0 to 600,000 KB/s) prior to standardization.
4. **Time-Grid Regularization**: Re-index timestamps to regular 300-second intervals to resolve the ~16% irregular deltas and gaps.
5. **Aggregation Strategy Choice**:
   - *VM-Level*: Model individual VM traces independently.
   - *Cluster-Level*: Aggregate across the 1,250 VMs at each 300-second timestamp to produce a cluster-wide time series directly comparable to Alibaba.

### 6.3 Technical Feasibility of 300-Second Aggregation
* **Yes, completely feasible**: Bitbrains is **already** collected at nominal 300-second intervals (median delta is 300 seconds; 83.61% exact). Aligning to a uniform 300s grid across all 1,250 VMs requires only basic forward-fill or linear interpolation for the occasional gap.

### 6.4 Recommended Usage in Research
We recommend treating Bitbrains under a **two-stage research roadmap**:

1. **Primary Recommendation: Separate Training & Validation Dataset (Benchmark Benchmark)**
   - Train an independent, dedicated Autoencoder (Dense or LSTM) directly on Bitbrains data using a dedicated scaler and 80/20 train/test split.
   - This proves whether the unsupervised autoencoder methodology generalizes to **large-scale individual VM telemetry** (11.2M rows across 1,250 machines) with heavy zero inflation and bursty I/O.
2. **Secondary Recommendation: Cross-Dataset Domain-Shift / Generalization Experiment**
   - After unit harmonization (e.g., standardizing or mapping throughput to utilization percentiles), evaluate cross-dataset performance.
   - Use the Alibaba model on Bitbrains to formally demonstrate and document **model drift, domain shift, and out-of-distribution breakdown** between cluster-level metrics and VM-level telemetry.
3. **Direct Zero-Shot Transfer**: **Not recommended** without domain adaptation, due to unit divergence (KB/s throughput vs percentage utilization).
