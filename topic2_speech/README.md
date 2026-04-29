# VE216 Project Topic 2：语音分析（MSWC English 精确首浊辅音主线）

当前主线已经收紧到 `MSWC English` 的四类精确首浊辅音任务。`report_cn.tex` 里采用的定义是：只保留首字母属于 `g/b/d/z`，且 CMUdict / ARPABET 中首个非元音音素严格对应 `G/B/D/Z` 的单词。最终数据集是每类 `75` 个词，共 `300` 个词、`6796` 条音频，按 train/dev/test 划分为 `5244/776/776`。

## 当前主线文件

- `report_cn.tex`：中文报告源文件。
- `sections/*.tex`：报告正文拆分后的各个 section 文件，`report_cn.tex` 只负责串联。
- `src/download_mswc_en.sh`：下载 `MSWC English` 原始元数据、split 和音频到 `data/raw/mswc_en/`。
- `src/prepare_mswc_initial_gbdz_subset.py`：生成精确首浊辅音子集，当前报告使用 `balanced-cap` 模式输出 `data/raw/mswc_en_gbdz_initial_balanced_75w_cap40_5_5/`。
- `src/run_mswc_course.py`：主实验脚本，先用 dev 选块长度，再把 `strict_course_fft` 的类别模板在 train+dev 上重估，随后给出 `course_filterbank_corr`、`mlp_ai` 和 `cnn_ai` 对照。
- `src/matlab_course_template_detector.m`：严格课内 MATLAB 检测器，只用短时 FFT 幅度谱和归一化相关。
- `src/matlab_course_spectrum_demo.m`：输出 `figures/course_matlab/example_short_time_spectra.png` 和 `example_score_matrix.png`。
- `src/matlab_case_demo.m`：逐条预览四个案例音频，并显示严格课内分数条形图。
- `references/mswc-course-gbdz-75w-cap40_5_5.md`：本次实验记录。
- `build/mswc_course_gbdz_75w_cap40_5_5/`：结果目录，包含 `summary.json`、`summary_metrics.csv`、`per_label_metrics.csv`、`thresholds.csv`、`example_predictions.csv`、`cnn_history.csv`、`course_template_bank.json`、`strict_course_template_bank.json`。
- `figures/mswc_course_gbdz_75w_cap40_5_5/`：报告图表目录。
- `figures/mswc_course_gbdz_75w_cap40_5_5/course_filterbank_*.png`：扩展模板法的正负模板差分和局部定位可视化。
- `figures/mswc_course_gbdz_75w_cap40_5_5/course_filterbank_corr/`：扩展模板法的分标签得分分布。
- `figures/course_matlab/`：MATLAB 示例图目录。
- `cases/README.md`：四个案例音频说明。

## 数据集规则

- 只保留首字母属于 `g/b/d/z` 的单词。
- 只保留首个非元音音素与首字母严格对应的单词。
- 最终每个标签 `75` 个词，测试集每类 `194` 条音频。
- 报告里的代表样本是 `get`、`boy`、`did`、`zero`，其中 `/b/` 用 `boy`。

## 环境准备

```bash
cd /home/fanghaotian/Desktop/src/216-project/topic2_speech
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

`prepare_mswc_initial_gbdz_subset.py` 会调用 `ffmpeg` 把选中的音频统一转成 `16 kHz` 单声道 WAV。

## 复现命令

```bash
cd /home/fanghaotian/Desktop/src/216-project/topic2_speech

# 1. 下载原始 MSWC English 数据
src/download_mswc_en.sh

# 2. 生成精确首浊辅音子集
.venv/bin/python src/prepare_mswc_initial_gbdz_subset.py \
  --out-root data/raw/mswc_en_gbdz_initial_balanced_75w_cap40_5_5 \
  --sampling-mode balanced-cap \
  --words-per-label 75 \
  --tmp-root data/tmp_extract_gbdz_initial_75w_cap40_5_5

# 3. 运行主实验
.venv/bin/python src/run_mswc_course.py \
  --manifest data/raw/mswc_en_gbdz_initial_balanced_75w_cap40_5_5/manifest.csv \
  --labels g b d z \
  --cache build/mswc_course_gbdz_75w_cap40_5_5/features_v1.npz \
  --out-dir build/mswc_course_gbdz_75w_cap40_5_5 \
  --fig-dir figures/mswc_course_gbdz_75w_cap40_5_5 \
  --report-path references/mswc-course-gbdz-75w-cap40_5_5.md \
  --epochs 25 \
  --cnn-batch-size 192 \
  --cnn-lr 0.001 \
  --cnn-weight-decay 0.0001 \
  --cnn-patience 8

# 4. 编译中文报告
latexmk -xelatex -interaction=nonstopmode -halt-on-error report_cn.tex
```

## 当前结果

| 方法 | label-wise acc. | exact-match acc. | macro-F1 | micro-F1 | samples-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 严格课内 FFT 模板法 | 0.6327 | 0.2655 | 0.2439 | 0.2655 | 0.2655 |
| 课外扩展模板法 | 0.7513 | 0.5026 | 0.4974 | 0.5026 | 0.5026 |
| MLP | 0.7423 | 0.4845 | 0.4848 | 0.4845 | 0.4845 |
| CNN | 0.7796 | 0.5593 | 0.5557 | 0.5593 | 0.5593 |

## 结论摘记

- 主线已经切到严格课内 FFT 模板法。
- 严格课内版现在不再只靠 4 个 case 模板，而是先在 dev 上选出 `5` 帧块，再把 train+dev 的同类样本重新平均成最终模板。
- 课外扩展版只作为对照，不再放在前面讲。
- `/z/` 最容易，爆破音之间仍然更容易混淆。
- 旧的多任务、旧数据入口和旧升级脚本已清理，不再作为当前项目入口。

## 备注

- 当前 `report_cn.tex` 的图表和案例说明以 `figures/mswc_course_gbdz_75w_cap40_5_5/` 和 `figures/course_matlab/` 为准。
- 如果只看当前报告，优先读 `report_cn.tex` 和 `references/mswc-course-gbdz-75w-cap40_5_5.md`。
