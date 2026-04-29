function [result, bank] = matlab_course_template_detector(wavPath, templateSource, templateLabels)
%MATLAB_COURSE_TEMPLATE_DETECTOR Detect voiced consonants using short-time FFT templates.
%
% This is the strict "course-method" version:
%   1) short-time segmentation
%   2) FFT magnitude spectrum
%   3) normalized cross-correlation with per-letter templates
%   4) max-score decision
%
% Usage:
%   result = matlab_course_template_detector(wavPath)
%   [result, bank] = matlab_course_template_detector(wavPath, templatePaths, labels)
%   result = matlab_course_template_detector(wavPath, bank)
%
% Default templates are built from the four case files:
%   get, boy, did, zero

cfg = default_cfg();

if nargin < 1 || isempty(wavPath)
    error('wavPath is required.');
end

if nargin < 2 || isempty(templateSource)
    [templatePaths, templateLabels] = default_template_set(fileparts(mfilename('fullpath')));
    bank = build_template_bank(templatePaths, templateLabels, cfg);
elseif isstruct(templateSource)
    bank = templateSource;
else
    if nargin < 3 || isempty(templateLabels)
        [~, templateLabels] = default_template_set(fileparts(mfilename('fullpath')));
    end
    bank = build_template_bank(templateSource, templateLabels, cfg);
end

[x, sr] = read_mono_audio(wavPath);
x = ensure_target_sr(x, sr, bank.cfg.target_sr);

[scores, bestStarts] = score_all_labels(x, bank, bank.cfg);
[bestScore, bestIdx] = max(scores);

result = struct();
result.wavPath = string(wavPath);
result.labels = string(bank.labels(:));
result.scores = scores(:);
result.bestLabel = string(bank.labels(bestIdx));
result.bestScore = bestScore;
result.bestStartFrame = bestStarts(bestIdx);
result.bestStartTime = (bestStarts(bestIdx) - 1) * bank.cfg.hop_s;
result.templateFreqHz = bank.freq_hz(:);
end

function cfg = default_cfg()
cfg = struct();
cfg.target_sr = 16000;
cfg.frame_ms = 20;
cfg.hop_ms = 10;
cfg.block_frames = 4;
cfg.nfft = 512;
cfg.hop_s = cfg.hop_ms / 1000;
end

function [templatePaths, templateLabels] = default_template_set(srcDir)
caseDir = fullfile(srcDir, '..', 'cases');
templatePaths = {
    fullfile(caseDir, 'case_g_get.wav')
    fullfile(caseDir, 'case_b_boy.wav')
    fullfile(caseDir, 'case_d_did.wav')
    fullfile(caseDir, 'case_z_zero.wav')
};
templateLabels = {'g', 'b', 'd', 'z'};
end

function bank = build_template_bank(templatePaths, templateLabels, cfg)
templatePaths = cellstr(string(templatePaths(:)));
templateLabels = cellstr(string(templateLabels(:)));
if numel(templatePaths) ~= numel(templateLabels)
    error('templatePaths and templateLabels must have the same length.');
end

nLabels = numel(templateLabels);
templates = zeros(nLabels, floor(cfg.nfft / 2) + 1);
templateStartFrames = zeros(nLabels, 1);
templateWords = strings(nLabels, 1);

for i = 1:nLabels
    [x, sr] = read_mono_audio(templatePaths{i});
    x = ensure_target_sr(x, sr, cfg.target_sr);
    [descriptor, bestStart] = select_max_energy_descriptor(x, cfg);
    templates(i, :) = descriptor;
    templateStartFrames(i) = bestStart;
    [~, stem, ~] = fileparts(templatePaths{i});
    templateWords(i) = string(stem);
end

bank = struct();
bank.labels = string(templateLabels(:));
bank.template_paths = string(templatePaths(:));
bank.template_words = templateWords;
bank.templates = templates;
bank.template_start_frames = templateStartFrames;
bank.freq_hz = (0:(floor(cfg.nfft / 2)))' * (cfg.target_sr / cfg.nfft);
bank.cfg = cfg;
end

function [scores, bestStarts] = score_all_labels(x, bank, cfg)
descriptors = all_candidate_descriptors(x, cfg);
nCandidates = size(descriptors.matrix, 1);
nLabels = size(bank.templates, 1);

scores = -inf(nLabels, 1);
bestStarts = ones(nLabels, 1);
for j = 1:nLabels
    template = bank.templates(j, :).';
    candidateScores = descriptors.matrix * template;
    [scores(j), bestIdx] = max(candidateScores);
    if isempty(bestIdx)
        bestIdx = 1;
    end
    if nCandidates > 0
        bestStarts(j) = descriptors.start_frames(bestIdx);
    end
end
end

function descriptors = all_candidate_descriptors(x, cfg)
frames = frame_signal(x, cfg);
nFrames = size(frames, 1);
nStarts = max(1, nFrames - cfg.block_frames + 1);
matrix = zeros(nStarts, floor(cfg.nfft / 2) + 1);
startFrames = zeros(nStarts, 1);

for s = 1:nStarts
    matrix(s, :) = block_descriptor(frames, s, cfg);
    startFrames(s) = s;
end

descriptors = struct();
descriptors.matrix = matrix;
descriptors.start_frames = startFrames;
end

function [descriptor, bestStart] = select_max_energy_descriptor(x, cfg)
frames = frame_signal(x, cfg);
nFrames = size(frames, 1);
nStarts = max(1, nFrames - cfg.block_frames + 1);
bestEnergy = -inf;
bestStart = 1;
descriptor = zeros(1, floor(cfg.nfft / 2) + 1);

for s = 1:nStarts
    block = frames(s:(s + cfg.block_frames - 1), :);
    blockEnergy = sum(block(:) .^ 2);
    if blockEnergy > bestEnergy
        bestEnergy = blockEnergy;
        bestStart = s;
        descriptor = block_descriptor(frames, s, cfg);
    end
end
end

function descriptor = block_descriptor(frames, startFrame, cfg)
block = frames(startFrame:(startFrame + cfg.block_frames - 1), :);
spec = abs(fft(block, cfg.nfft, 2));
spec = spec(:, 1:(floor(cfg.nfft / 2) + 1));
rowMax = max(spec, [], 2);
rowMax(rowMax == 0) = 1;
spec = spec ./ rowMax;
descriptor = mean(spec, 1);
descriptor = descriptor ./ max(norm(descriptor), 1e-12);
end

function frames = frame_signal(x, cfg)
frameLen = max(1, round(cfg.target_sr * cfg.frame_ms / 1000));
hop = max(1, round(cfg.target_sr * cfg.hop_ms / 1000));
if numel(x) < frameLen
    x = [x; zeros(frameLen - numel(x), 1)];
end
win = periodic_hamming(frameLen);
frameCount = floor((numel(x) - frameLen) / hop) + 1;
frames = zeros(frameCount, frameLen);
for i = 1:frameCount
    idx = (i - 1) * hop + (1:frameLen);
    frames(i, :) = x(idx).' .* win.';
end
end

function win = periodic_hamming(n)
if n <= 1
    win = ones(max(1, n), 1);
    return;
end
k = (0:(n - 1))';
win = 0.54 - 0.46 * cos(2 * pi * k / n);
end

function [x, sr] = read_mono_audio(wavPath)
[x, sr] = audioread(wavPath);
if size(x, 2) > 1
    x = mean(x, 2);
end
x = double(x(:));
peak = max(abs(x));
if peak > 0
    x = x / peak;
end
end

function x = ensure_target_sr(x, sr, targetSr)
if sr == targetSr
    return;
end
if exist('resample', 'file') == 2
    x = resample(x, targetSr, sr);
else
    t = (0:numel(x)-1)' / sr;
    tNew = (0:1/targetSr:t(end))';
    x = interp1(t, x, tNew, 'linear', 'extrap');
end
end
