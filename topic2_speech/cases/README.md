# Case Audio

This folder contains four representative case clips for the report and MATLAB demo.

- `case_g_get.wav` -> `/g/` case, source word `get`
- `case_b_boy.wav` -> `/b/` case, source word `boy`
- `case_d_did.wav` -> `/d/` case, source word `did`
- `case_z_zero.wav` -> `/z/` case, source word `zero`

`book` is not present in the current MSWC subset, so `boy` is used as the /b/ example.

Files:

- `case_manifest.csv`: source mapping for the four clips
- `course_template_bank.json`: template bank exported from the tuned course detector

MATLAB entry point:

- `../src/matlab_case_demo.m`
- `../src/matlab_letter_detector.m`
