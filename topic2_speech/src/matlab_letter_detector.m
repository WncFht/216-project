function result = matlab_letter_detector(wavPath, bankPath, targetLabels)
%MATLAB_LETTER_DETECTOR Detect voiced consonants with the tuned course template method.
%
% result = matlab_letter_detector(wavPath)
% result = matlab_letter_detector(wavPath, bankPath)
% result = matlab_letter_detector(wavPath, bankPath, targetLabels)
%
% Pipeline:
%   waveform -> pre-emphasis -> short-time FFT -> 40-band log-mel spectrum
%   -> voiced onset detection -> local patch features
%   -> positive/negative template contrast + envelope autocorrelation
%   -> threshold or exclusive decoding

if nargin < 2 || isempty(bankPath)
    bankPath = resolve_template_bank_path(fileparts(mfilename('fullpath')));
else
    bankPath = string(bankPath);
    if isfolder(bankPath)
        bankPath = fullfile(bankPath, 'course_template_bank.json');
    end
end
if ~isfile(bankPath)
    bankPath = resolve_template_bank_path(fileparts(mfilename('fullpath')));
end
if ~isfile(bankPath)
    error('Cannot find course_template_bank.json. Provide bankPath explicitly or place the file next to the script.');
end

if nargin < 3 || isempty(targetLabels)
    targetLabels = strings(0, 1);
else
    targetLabels = string(targetLabels(:));
end

bank = jsondecode(fileread(bankPath));
labels = string(bank.labels(:));
targetMask = true(size(labels));
if ~isempty(targetLabels)
    targetMask = ismember(labels, targetLabels);
end

[x, sr] = audioread(wavPath);
if size(x, 2) > 1
    x = mean(x, 2);
end
x = double(x(:));
peak = max(abs(x));
if peak > 0
    x = x / peak;
end

targetSr = double(bank.target_sr);
if sr ~= targetSr
    if exist('resample', 'file') == 2
        x = resample(x, targetSr, sr);
    else
        t = (0:numel(x)-1)' / sr;
        tNew = (0:1/targetSr:t(end))';
        x = interp1(t, x, tNew, 'linear', 'extrap');
    end
    sr = targetSr;
end

feat = compute_course_features( ...
    x, ...
    sr, ...
    double(bank.frame_ms), ...
    double(bank.hop_ms), ...
    double(bank.n_mels), ...
    double(bank.autocorr_lags), ...
    double(bank.window_frames), ...
    double(bank.pre_frames), ...
    double(bank.segment_count), ...
    double(bank.energy_smooth_frames), ...
    double(bank.onset_margin));

patchPosTemplates = double(resolve_bank_field(bank, {'patch_pos_templates', 'patch_templates', 'spectrum_templates'}));
patchNegTemplates = double(resolve_bank_field(bank, {'patch_neg_templates'}));
autocorrPosTemplates = double(resolve_bank_field(bank, {'autocorr_pos_templates', 'autocorr_templates'}));
autocorrNegTemplates = double(resolve_bank_field(bank, {'autocorr_neg_templates'}));
thresholds = double(bank.thresholds(:));
crossWeight = double(bank.cross_correlation_weight);
autocorrWeight = double(bank.autocorrelation_weight);
candidateShifts = double(bank.candidate_shifts(:)');
exclusiveDecode = false;
if isfield(bank, 'exclusive_decode')
    exclusiveDecode = logical(bank.exclusive_decode);
end

[patchRows, autoRows] = extract_course_candidates( ...
    feat.logmel, ...
    feat.onset_idx, ...
    double(bank.window_frames), ...
    double(bank.pre_frames), ...
    double(bank.segment_count), ...
    double(bank.autocorr_lags), ...
    candidateShifts);

scores = zeros(numel(labels), 1);
for j = 1:numel(labels)
    patchPosScores = cosine_similarity_rows(patchRows, patchPosTemplates(j, :));
    patchNegScores = cosine_similarity_rows(patchRows, patchNegTemplates(j, :));
    autoPosScores = cosine_similarity_rows(autoRows, autocorrPosTemplates(j, :));
    autoNegScores = cosine_similarity_rows(autoRows, autocorrNegTemplates(j, :));
    candidateScores = crossWeight * (patchPosScores - patchNegScores) + autocorrWeight * (autoPosScores - autoNegScores);
    scores(j) = max(candidateScores);
end

if exclusiveDecode
    predMask = false(size(scores));
    [bestScore, bestIdx] = max(scores);
    predMask(bestIdx) = true;
else
    predMask = scores >= thresholds;
    [bestScore, bestIdx] = max(scores);
end

result = struct();
result.wavPath = string(wavPath);
result.bankPath = string(bankPath);
result.labels = labels;
result.scores = scores;
result.thresholds = thresholds;
result.predMask = predMask;
result.predLabels = labels(predMask);
result.bestLabel = labels(bestIdx);
result.bestScore = bestScore;
result.targetLabels = labels(targetMask);
result.targetScores = scores(targetMask);
result.features = feat;
result.exclusiveDecode = exclusiveDecode;
end

function feat = compute_course_features(x, sr, frameMs, hopMs, nMels, nLags, windowFrames, preFrames, segmentCount, smoothFrames, onsetMargin)
x = pre_emphasis(x, 0.97);
frames = frame_signal(x, sr, frameMs, hopMs);
if isempty(frames)
    frames = zeros(1, max(1, round(sr * frameMs / 1000)));
end

nFft = 512;
spec = abs(fft(frames, nFft, 2));
spec = spec(:, 1:(floor(nFft / 2) + 1));
power = (spec .^ 2) / nFft;
fb = mel_filterbank(sr, nFft, nMels, 50, sr / 2);
logmel = log(max(power * fb.', 1e-12));

frameEnergy = frame_energy_from_logmel(logmel);
smoothedEnergy = moving_average_edge(frameEnergy, smoothFrames);
onsetIdx = detect_course_onset(smoothedEnergy, onsetMargin);

feat = struct();
feat.logmel = logmel;
feat.frame_energy = frameEnergy;
feat.smoothed_energy = smoothedEnergy;
feat.onset_idx = onsetIdx;
feat.window_frames = windowFrames;
feat.pre_frames = preFrames;
feat.segment_count = segmentCount;
feat.autocorr_lags = nLags;
end

function y = pre_emphasis(x, alpha)
x = double(x(:));
y = filter([1, -alpha], 1, x);
end

function frames = frame_signal(x, sr, frameMs, hopMs)
frameLen = max(1, round(sr * frameMs / 1000));
hop = max(1, round(sr * hopMs / 1000));
x = double(x(:));
if numel(x) < frameLen
    x = [x; zeros(frameLen - numel(x), 1)];
end
nFrames = floor((numel(x) - frameLen) / hop) + 1;
frames = zeros(nFrames, frameLen);
win = hamming(frameLen, 'periodic');
for i = 1:nFrames
    idx = (i - 1) * hop + (1:frameLen);
    frames(i, :) = x(idx).' .* win.';
end
end

function fb = mel_filterbank(sr, nFft, nMels, fmin, fmax)
if nargin < 4 || isempty(fmin)
    fmin = 50;
end
if nargin < 5 || isempty(fmax)
    fmax = sr / 2;
end
nBins = floor(nFft / 2) + 1;
fb = zeros(nMels, nBins);
melEdges = linspace(hz_to_mel(fmin), hz_to_mel(fmax), nMels + 2);
hzEdges = mel_to_hz(melEdges);
bins = floor((nFft + 1) * hzEdges / sr);
for m = 1:nMels
    left = max(bins(m), 0);
    center = max(bins(m + 1), left + 1);
    right = max(bins(m + 2), center + 1);
    left = min(left, nBins - 1);
    center = min(center, nBins - 1);
    right = min(right, nBins - 1);
    for k = left:center
        if center > left
            fb(m, k + 1) = (k - left) / (center - left);
        end
    end
    for k = center:right
        if right > center
            fb(m, k + 1) = (right - k) / (right - center);
        end
    end
end
rowSums = sum(fb, 2);
rowSums(rowSums == 0) = 1;
fb = fb ./ rowSums;
end

function mel = hz_to_mel(hz)
mel = 2595 * log10(1 + hz / 700);
end

function hz = mel_to_hz(mel)
hz = 700 * (10 .^ (mel / 2595) - 1);
end

function energy = frame_energy_from_logmel(logmel)
energy = log(mean(exp(logmel), 2) + 1e-12);
end

function y = moving_average_edge(x, width)
x = double(x(:));
width = max(1, round(width));
if width <= 1 || isempty(x)
    y = x;
    return;
end
padLeft = floor(width / 2);
padRight = width - 1 - padLeft;
padded = [repmat(x(1), padLeft, 1); x; repmat(x(end), padRight, 1)];
kernel = ones(width, 1) / width;
y = conv(padded, kernel, 'valid');
end

function onsetIdx = detect_course_onset(smoothedEnergy, onsetMargin)
headCount = min(numel(smoothedEnergy), max(6, 10));
baseline = median(smoothedEnergy(1:headCount));
cutoff = baseline + onsetMargin;
idx = find(smoothedEnergy >= cutoff, 1, 'first');
if isempty(idx)
    [~, idx] = max(smoothedEnergy);
end
onsetIdx = idx;
end

function vec = segment_mean_spectrum(patch, segmentCount)
parts = round(linspace(1, size(patch, 1) + 1, segmentCount + 1));
rows = cell(segmentCount, 1);
for i = 1:segmentCount
    s = parts(i);
    e = max(parts(i + 1) - 1, s);
    rows{i} = mean(patch(s:e, :), 1);
end
vec = reshape(cat(2, rows{:}), 1, []);
end

function patch = pad_or_trim_rows(x, startIdx, lengthWanted)
nRows = size(x, 1);
nCols = size(x, 2);
if nRows == 0
    patch = zeros(lengthWanted, nCols);
    return;
end
startIdx = round(startIdx);
endIdx = startIdx + lengthWanted - 1;
if startIdx > nRows
    patch = repmat(x(end, :), lengthWanted, 1);
    return;
end
if endIdx < 1
    patch = repmat(x(1, :), lengthWanted, 1);
    return;
end
left = max(0, 1 - startIdx);
right = max(0, endIdx - nRows);
s = max(1, startIdx);
e = min(nRows, endIdx);
if s > e
    patch = repmat(x(min(max(startIdx, 1), nRows), :), lengthWanted, 1);
    return;
end
patch = x(s:e, :);
if left > 0
    patch = [repmat(patch(1, :), left, 1); patch];
end
if right > 0
    patch = [patch; repmat(patch(end, :), right, 1)];
end
if size(patch, 1) > lengthWanted
    patch = patch(1:lengthWanted, :);
elseif size(patch, 1) < lengthWanted
    patch = [patch; repmat(patch(end, :), lengthWanted - size(patch, 1), 1)];
end
end

function [patchVec, patchAuto] = extract_course_patch(logmel, onsetIdx, windowFrames, preFrames, segmentCount, nLags, shift)
startIdx = onsetIdx - preFrames + shift;
patch = pad_or_trim_rows(logmel, startIdx, windowFrames);
if preFrames > 0
    baseline = mean(patch(1:preFrames, :), 1);
else
    baseline = patch(1, :);
end
centeredPatch = patch - baseline;
rawSegment = segment_mean_spectrum(patch, segmentCount);
centeredSegment = segment_mean_spectrum(centeredPatch, segmentCount);
patchVec = [rawSegment, centeredSegment];
patchAuto = normalized_autocorrelation(frame_energy_from_logmel(patch), nLags);
end

function [patchRows, autoRows] = extract_course_candidates(logmel, onsetIdx, windowFrames, preFrames, segmentCount, nLags, shifts)
nShifts = numel(shifts);
samplePatch = extract_course_patch(logmel, onsetIdx, windowFrames, preFrames, segmentCount, nLags, 0);
patchDim = numel(samplePatch);
patchRows = zeros(nShifts, patchDim);
autoRows = zeros(nShifts, nLags);
for i = 1:nShifts
    [patchVec, patchAuto] = extract_course_patch(logmel, onsetIdx, windowFrames, preFrames, segmentCount, nLags, shifts(i));
    patchRows(i, :) = patchVec;
    autoRows(i, :) = patchAuto;
end
end

function values = normalized_autocorrelation(x, nLags)
x = double(x(:));
x = x - mean(x);
denom = dot(x, x) + eps;
values = zeros(1, nLags);
if isempty(x)
    return;
end
for lag = 0:(nLags - 1)
    if lag >= numel(x)
        break;
    end
    values(lag + 1) = dot(x(1:end-lag), x(1+lag:end)) / denom;
end
end

function scores = cosine_similarity_rows(X, template)
X = double(X);
template = double(template(:)).';
scores = zeros(size(X, 1), 1);
templateNorm = norm(template) + eps;
for i = 1:size(X, 1)
    row = X(i, :);
    scores(i) = dot(row, template) / ((norm(row) + eps) * templateNorm);
end
end

function value = resolve_bank_field(bank, fieldNames)
for i = 1:numel(fieldNames)
    if isfield(bank, fieldNames{i})
        value = bank.(fieldNames{i});
        return;
    end
end
error('Missing required field in course_template_bank.json: %s', strjoin(string(fieldNames), ', '));
end

function bankPath = resolve_template_bank_path(startDir)
candidateDirs = {
    startDir
    fullfile(startDir, 'cases')
    fullfile(startDir, 'case')
    fullfile(startDir, '..', 'cases')
    fullfile(startDir, '..', 'case')
    fullfile(startDir, '..', 'topic2_speech', 'cases')
    fullfile(startDir, '..', 'topic2_speech', 'case')
};
for i = 1:numel(candidateDirs)
    candidate = fullfile(candidateDirs{i}, 'course_template_bank.json');
    if isfile(candidate)
        bankPath = candidate;
        return;
    end
end
error('Unable to locate course_template_bank.json in any known layout.');
end
