# VE216 Project Topic 2：语音分析（MSWC English 精确首浊辅音主线）

当前目录的主线已经切到 `MSWC English` 的四类精确首浊辅音任务。`report_cn.tex` 使用的定义是：只保留首字母属于 `g/b/d/z`，且 CMUdict / ARPABET 中首个非元音音素严格对应 `G/B/D/Z` 的单词。最终数据集是 `60` 个词、`3000` 条音频，按 `40/5/5` 划分为 `2400/300/300`。

## 当前主线文件

- `report_cn.tex`：中文报告源文件。
- `src/download_mswc_en.sh`：下载 `MSWC English` 原始元数据、split 和音频到 `data/raw/mswc_en/`。
- `src/prepare_mswc_initial_gbdz_subset.py`：生成精确首浊辅音子集，默认输出 `data/raw/mswc_en_gbdz_initial_balanced_40_5_5/`，并写出 `selection_summary.json`、`eligible_word_pool.csv`、`manifest.csv`、`word_summary.csv`、`label_summary.csv`。
- `src/run_mswc_course.py`：在该子集上跑课程模板法、MLP、CNN；报告对应 `--labels g b d z`。
- `references/mswc-course-gbdz-local_patch_v1.md`：当前报告图表对应的实验记录。
- `references/mswc-course-gbdz-initial-baseline.md`：同一数据集上的基线记录。
- `build/mswc_course_gbdz_local_patch_v1/`：报告用结果目录，包含 `summary.json`、`summary_metrics.csv`、`per_label_metrics.csv`、`thresholds.csv`、`example_predictions.csv`、`cnn_history.csv`、`course_template_bank.json`。
- `figures/mswc_course_gbdz_local_patch_v1/`：报告用图表目录。
- `cases/README.md`：四个案例音频说明。

## 数据集规则

- 只保留首字母属于 `g/b/d/z` 的单词。
- 只保留首个非元音音素与首字母严格对应的单词。
- 最终每个标签 `15` 个词，测试集每类 `75` 条音频。
- 报告里的代表样本是 `get`、`boy`、`did`、`zero`，其中 `/b/` 用 `boy`，因为当前子集里没有 `book`。

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
.venv/bin/python src/prepare_mswc_initial_gbdz_subset.py

# 3. 运行报告对应的 local_patch_v1 配置
.venv/bin/python src/run_mswc_course.py \
  --manifest data/raw/mswc_en_gbdz_initial_balanced_40_5_5/manifest.csv \
  --labels g b d z \
  --cache build/mswc_course_gbdz_local_patch_v1/features_v1.npz \
  --out-dir build/mswc_course_gbdz_local_patch_v1 \
  --fig-dir figures/mswc_course_gbdz_local_patch_v1 \
  --report-path references/mswc-course-gbdz-local_patch_v1.md \
  --epochs 10 \
  --cnn-batch-size 192 \
  --cnn-lr 0.001 \
  --cnn-weight-decay 0.0001 \
  --cnn-patience 5

# 4. 编译中文报告
latexmk -xelatex -interaction=nonstopmode -halt-on-error report_cn.tex
```

## 当前结果

| 方法 | label-wise acc. | exact-match acc. | macro-F1 | micro-F1 | samples-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 课程模板法 | 0.7750 | 0.5500 | 0.5462 | 0.5500 | 0.5500 |
| MLP | 0.7633 | 0.5267 | 0.5258 | 0.5267 | 0.5267 |
| CNN | 0.6600 | 0.3200 | 0.2219 | 0.3200 | 0.3200 |

## 结论摘记

- `/g/` 最难。
- `/z/` 最容易。
- `/b/` 提升最明显。
- `/d/` 居中。

## 其他脚本

- `src/download_data.sh`：旧 `mini_speech_commands` 数据入口。
- `src/download_librispeech_subset.sh`：旧 `LibriSpeech` 入口。
- `src/prepare_mswc_subset.py`：旧 600 词子集脚本，连同 `data/raw/mswc_en_subset_600/`、`build/mswc_course_600/`、`figures/mswc_course_600/`、`references/mswc-course-results-600.md` 一起保留作历史对照。

## 备注

- 当前 `report_cn.tex` 的图表和案例说明以 `figures/mswc_course_gbdz_local_patch_v1/` 为准。
- 如果只看当前报告，优先读 `report_cn.tex` 和 `references/mswc-course-gbdz-local_patch_v1.md`。
