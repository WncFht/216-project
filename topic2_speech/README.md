# VE216 Project Topic 2：语音分析（MSWC English 600 词主线）

本目录当前已经完成两件事：

1. 把数据主线切到 `MSWC English` 的 `600` 词子集；
2. 在这批数据上跑通了 `课内方法 + MLP + CNN` 的同口径对比。

当前仓库里最应该优先引用的闭环是：

> `600词 MSWC English -> 短时频谱 / filter bank -> course detector / MLP / CNN -> dev 调阈值 -> test 指标 -> 图表 / markdown / 中文报告`

## 当前主线文件

- `handoff.md`：本轮交接说明与任务边界。
- `report_cn.tex`：中文报告 LaTeX 源文件。
- `report_cn.pdf`：编译后的中文报告。
- `src/download_mswc_en.sh`：下载官方 `MSWC English` 原始数据。
- `src/prepare_mswc_subset.py`：子集准备脚本；默认值已切到 `600 / 40 / 5 / 5`，支持 `--seed-words-per-label`。
- `src/run_mswc_course.py`：600 词主实验脚本；默认读 `data/raw/mswc_en_subset_600/manifest.csv`，默认输出到 `build/mswc_course_600/` 和 `figures/mswc_course_600/`。
- `data/raw/mswc_en_subset_600/`：新的 600 词固定子集。
- `build/mswc_course_600/`：600 词主实验结果输出。
- `figures/mswc_course_600/`：600 词主线图表。
- `references/mswc-course-results-600.md`：自动生成的实验记录。
- `references/mswc_dataset_split_note.tex`：为什么从 160 词迁移到 600 词的设计说明。

## 当前已验证的数据规模

当前已经准备好的 `MSWC English` 新子集为：

- `600` 个英文单词；
- `30000` 条音频；
- `train/dev/test = 24000/3000/3000`；
- 每词严格 `40/5/5`；
- 全部统一为 `16 kHz` 单声道 WAV；
- `seed_words_per_label = 20`；
- 四个主标签 `/b d l n/` 的词级覆盖分别为 `62 / 159 / 169 / 211`。

旧版 `160` 词子集仍保留在 `data/raw/mswc_en_subset/`，只作为历史对照。

## 当前已验证的 600 词对比结果

执行命令：

```bash
cd 216/project/topic2_speech
.venv/bin/python src/run_mswc_course.py --epochs 20
```

结果来自：

- `build/mswc_course_600/summary_metrics.csv`
- `build/mswc_course_600/per_label_metrics.csv`
- `build/mswc_course_600/thresholds.csv`
- `build/mswc_course_600/cnn_history.csv`

### 整体指标

| method | label-wise acc. | exact-match acc. | macro-F1 | micro-F1 |
| --- | ---: | ---: | ---: | ---: |
| course_filterbank_corr | 0.3082 | 0.0167 | 0.3935 | 0.4104 |
| mlp_ai | 0.6181 | 0.1553 | 0.4586 | 0.4840 |
| cnn_ai | 0.6869 | 0.2287 | 0.4850 | 0.5167 |

### 分字母 F1

| label | course | MLP | CNN |
| --- | ---: | ---: | ---: |
| /b/ | 0.1902 | 0.2289 | 0.3217 |
| /d/ | 0.4209 | 0.4415 | 0.4715 |
| /l/ | 0.4421 | 0.5538 | 0.5385 |
| /n/ | 0.5208 | 0.6102 | 0.6083 |

### 当前结论

- `CNN` 拿到了最好的整体指标：label-wise accuracy、exact-match accuracy、macro-F1、micro-F1 都是当前最好。
- `MLP` 明显优于课内方法，而且在 `/l/`、`/n/` 两个标签上略强于 `CNN`。
- 课内方法仍然是最可解释的基线，但表现出明显的“高召回、低精度、误报偏多”。
- `/b/` 仍然是最难的标签，`CNN` 虽然也没有完全解决，但已经把 F1 从 `0.190` 提高到 `0.322`。

## 复现主线

```bash
cd 216/project/topic2_speech

uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 如果本地还没有 MSWC English 原始数据：
src/download_mswc_en.sh

# 生成 600 词子集
.venv/bin/python src/prepare_mswc_subset.py \
  --out-root data/raw/mswc_en_subset_600 \
  --target-words 600 \
  --train-per-word 40 \
  --dev-per-word 5 \
  --test-per-word 5 \
  --seed-words-per-label 20

# 跑完整对比：课程基线 + MLP + CNN
.venv/bin/python src/run_mswc_course.py \
  --manifest data/raw/mswc_en_subset_600/manifest.csv \
  --cache build/mswc_course_600/features_v1.npz \
  --out-dir build/mswc_course_600 \
  --fig-dir figures/mswc_course_600 \
  --report-path references/mswc-course-results-600.md \
  --epochs 20

# 只跑课程基线时可加：
#   --course-only

# 编译中文报告
latexmk -xelatex -interaction=nonstopmode -halt-on-error report_cn.tex
```

## 当前主线产物

数据与结果：

- `data/raw/mswc_en_subset_600/selection_summary.json`
- `data/raw/mswc_en_subset_600/manifest.csv`
- `data/raw/mswc_en_subset_600/word_summary.csv`
- `build/mswc_course_600/summary.json`
- `build/mswc_course_600/summary_metrics.csv`
- `build/mswc_course_600/per_label_metrics.csv`
- `build/mswc_course_600/thresholds.csv`
- `build/mswc_course_600/example_predictions.csv`
- `build/mswc_course_600/cnn_history.csv`
- `build/mswc_course_600/course_template_bank.json`
- `references/mswc-course-results-600.md`

图表：

- `figures/mswc_course_600/letter_examples.png`
- `figures/mswc_course_600/template_grid.png`
- `figures/mswc_course_600/metric_summary.png`
- `figures/mswc_course_600/per_label_f1.png`
- `figures/mswc_course_600/score_distribution_b.png`
- `figures/mswc_course_600/score_distribution_d.png`
- `figures/mswc_course_600/score_distribution_l.png`
- `figures/mswc_course_600/score_distribution_n.png`
- `figures/mswc_course_600/cnn_history.png`

## 历史文件与旧语境

下面这些内容还保留着，但已经不是当前主线：

- `data/raw/mswc_en_subset/`：旧 160 词子集。
- `build/mswc_course/`、`figures/mswc_course/`：旧 160 词主实验产物。
- `references/mswc-course-results.md`：旧 160 词结果说明。
- `handoff_legacy_160words.md`：旧版交接。
- `src/run_experiment.py`
- `src/run_upgrade.py`
- `src/run_multilabel_course.py`
- `src/run_librispeech_course.py`
- `report.tex`

## 当前仍未完成的边界

下面这些内容在当前版本里仍然没有完成：

- MATLAB / Octave 运行级验证；
- 针对 `book / boy / dog` 的定向词表设计；
- phone-level segmentation 或字母边界对齐；
- 把 600 词 AI 对比进一步扩到更多浊辅音标签。
