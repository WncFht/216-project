function result = matlab_letter_detector(wavPath, bankPath, targetLabels)
%MATLAB_LETTER_DETECTOR Detect voiced consonants with the course template method.
%
% result = matlab_letter_detector(wavPath)
% result = matlab_letter_detector(wavPath, bankPath)
% result = matlab_letter_detector(wavPath, bankPath, targetLabels)
%
% The pipeline matches the report:
%   waveform -> pre-emphasis -> short-time FFT -> 40-band log-mel spectrum
%   -> mean spectrum template + envelope autocorrelation
%   -> normalized cross-correlation + cosine similarity
%   -> per-label thresholding from dev data.

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
    double(bank.autocorr_lags));

spectrumTemplates = double(bank.spectrum_templates);
autocorrTemplates = double(bank.autocorr_templates);
thresholds = double(bank.thresholds(:));
crossWeight = double(bank.cross_correlation_weight);
autocorrWeight = double(bank.autocorrelation_weight);

scores = zeros(numel(labels), 1);
for j = 1:numel(labels)
    crossScore = max_normalized_xcorr(feat.course_spectrum, spectrumTemplates(j, :));
    autoScore = cosine_similarity(feat.course_autocorr, autocorrTemplates(j, :));
    scores(j) = crossWeight * crossScore + autocorrWeight * autoScore;
end

predMask = scores >= thresholds;
[bestScore, bestIdx] = max(scores);

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
end

function feat = compute_course_features(x, sr, frameMs, hopMs, nMels, nLags)
x = pre_emphasis(x, 0.97);
frames = frame_signal(x, sr, frameMs, hopMs);
if isempty(frames)
    frames = x(:).';
end

frameLen = size(frames, 2);
if frameLen == 1
    win = 1;
else
    n = 0:(frameLen - 1);
    win = 0.54 - 0.46 * cos(2 * pi * n / (frameLen - 1));
end
frames = bsxfun(@times, frames, win);

nFft = 512;
spec = abs(fft(frames, nFft, 2));
spec = spec(:, 1:(nFft / 2 + 1));
power = (spec .^ 2) / nFft;

fb = mel_filterbank(sr, nFft, nMels, 50, sr / 2);
logmel = log(max(power * fb.', 1e-12));

courseSpectrum = mean(logmel, 1);
courseEnvelope = mean(logmel, 2);
courseAutocorr = normalized_autocorrelation(courseEnvelope, nLags);

feat = struct();
feat.logmel = logmel;
feat.course_spectrum = courseSpectrum;
feat.course_autocorr = courseAutocorr;
feat.frame_envelope = courseEnvelope;
end

function x = pre_emphasis(x, coeff)
x = x(:);
if isempty(x)
    return;
end
x = [x(1); x(2:end) - coeff * x(1:end-1)];
end

function frames = frame_signal(x, sr, frameMs, hopMs)
frameLen = max(1, round(sr * frameMs / 1000));
hop = max(1, round(sr * hopMs / 1000));
x = x(:);
if numel(x) < frameLen
    x = [x; zeros(frameLen - numel(x), 1)];
end
nFrames = 1 + floor((numel(x) - frameLen) / hop);
frames = zeros(nFrames, frameLen);
for i = 1:nFrames
    idx = (1:frameLen) + (i - 1) * hop;
    frames(i, :) = x(idx).';
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
    right = min(right, nBins - 1);
    center = min(center, nBins - 1);

    for k = left:center
        if k >= 0 && k < nBins
            fb(m, k + 1) = (k - left) / max(center - left, 1);
        end
    end
    for k = center:right
        if k >= 0 && k < nBins
            fb(m, k + 1) = (right - k) / max(right - center, 1);
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

function values = normalized_autocorrelation(x, nLags)
x = double(x(:));
x = x - mean(x);
denom = dot(x, x) + eps;
values = zeros(1, nLags);
maxLag = min(nLags, numel(x));
for lag = 0:maxLag - 1
    if lag == 0
        values(lag + 1) = dot(x, x) / denom;
    else
        values(lag + 1) = dot(x(1:end - lag), x(1 + lag:end)) / denom;
    end
end
end

function score = max_normalized_xcorr(row, template)
row = double(row(:));
template = double(template(:));
row = row - mean(row);
template = template - mean(template);
denom = norm(row) * norm(template) + eps;
score = dot(row, template) / denom;
end

function score = cosine_similarity(x, y)
x = double(x(:));
y = double(y(:));
denom = norm(x) * norm(y) + eps;
score = dot(x, y) / denom;
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
