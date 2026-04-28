function result = matlab_letter_detector(wavPath, bankPath, targetLabels)
% MATLAB-style voiced-consonant detector for VE216 Topic 2.
% The bank is exported by src/run_mswc_course.py and uses:
% filter-bank spectrum template + autocorrelation template + thresholding.

    if nargin < 2 || isempty(bankPath)
        bankPath = fullfile('build', 'mswc_course_600', 'course_template_bank.json');
    end

    bank = jsondecode(fileread(bankPath));
    labels = string(bank.labels);
    if nargin < 3 || isempty(targetLabels)
        targetMask = true(size(labels));
    else
        targetLabels = string(targetLabels);
        targetMask = ismember(labels, targetLabels);
    end

    [x, sr] = audioread(wavPath);
    [spectrumVec, autocorrVec] = compute_course_features( ...
        x, ...
        sr, ...
        double(bank.target_sr), ...
        double(bank.frame_ms), ...
        double(bank.hop_ms), ...
        double(bank.n_mels), ...
        double(bank.autocorr_lags) ...
    );

    spectrumTemplates = double(bank.spectrum_templates);
    autocorrTemplates = double(bank.autocorr_templates);
    thresholds = double(bank.thresholds(:));
    crossWeight = double(bank.cross_correlation_weight);
    autocorrWeight = double(bank.autocorrelation_weight);

    scores = zeros(numel(labels), 1);
    for j = 1:numel(labels)
        crossScore = max_normalized_xcorr(spectrumVec, spectrumTemplates(j, :)');
        autocorrScore = cosine_similarity(autocorrVec, autocorrTemplates(j, :)');
        scores(j) = crossWeight * crossScore + autocorrWeight * autocorrScore;
    end

    predMask = scores >= thresholds;
    predLabels = labels(predMask);

    result = struct();
    result.labels = labels(targetMask);
    result.scores = scores(targetMask);
    result.thresholds = thresholds(targetMask);
    result.predMask = predMask(targetMask);
    result.predLabels = predLabels(ismember(predLabels, labels(targetMask)));
    result.bankPath = string(bankPath);
end


function [spectrumVec, autocorrVec] = compute_course_features(x, sr, targetSr, frameMs, hopMs, nMels, nLags)
    if size(x, 2) > 1
        x = mean(x, 2);
    end
    x = single(x(:));
    peak = max(abs(x));
    if peak > 0
        x = x ./ peak;
    end

    if sr ~= targetSr
        x = resample(x, targetSr, sr);
        sr = targetSr;
    end
    if numel(x) < targetSr
        x = [x; zeros(targetSr - numel(x), 1, 'single')];
    elseif numel(x) > targetSr
        x = x(1:targetSr);
    end

    frameLen = round(sr * frameMs / 1000);
    hop = round(sr * hopMs / 1000);
    if numel(x) < frameLen
        x = [x; zeros(frameLen - numel(x), 1, 'single')];
    end
    nFrames = 1 + floor((numel(x) - frameLen) / hop);
    idx = (0:frameLen-1)' + (0:nFrames-1) * hop + 1;
    frames = x(idx) .* hamming(frameLen, 'periodic');

    nFft = 512;
    spec = fft(frames, nFft, 1);
    mag = abs(spec(1:nFft/2+1, :))';
    power = (mag .^ 2) / nFft;
    fb = mel_filterbank(sr, nFft, nMels, 50, sr / 2);
    logmel = log(max(power * fb', 1e-12));

    spectrumVec = mean(logmel, 1)';
    envelope = mean(logmel, 2);
    autocorrVec = normalized_autocorrelation(envelope, nLags);
end


function fb = mel_filterbank(sr, nFft, nMels, fmin, fmax)
    mels = linspace(hz_to_mel(fmin), hz_to_mel(fmax), nMels + 2);
    hz = mel_to_hz(mels);
    bins = floor((nFft + 1) * hz / sr);
    fb = zeros(nMels, nFft / 2 + 1);
    for m = 2:nMels+1
        left = bins(m - 1);
        center = bins(m);
        right = bins(m + 1);
        if center <= left
            center = left + 1;
        end
        if right <= center
            right = center + 1;
        end
        for k = left:min(center - 1, size(fb, 2) - 1)
            fb(m - 1, k + 1) = (k - left) / max(center - left, 1);
        end
        for k = center:min(right - 1, size(fb, 2) - 1)
            fb(m - 1, k + 1) = (right - k) / max(right - center, 1);
        end
    end
end


function value = hz_to_mel(hz)
    value = 2595 * log10(1 + hz / 700);
end


function value = mel_to_hz(mel)
    value = 700 * (10 .^ (mel / 2595) - 1);
end


function values = normalized_autocorrelation(x, nLags)
    x = double(x(:));
    x = x - mean(x);
    denom = dot(x, x) + 1e-12;
    corr = xcorr(x, x, nLags - 1, 'none');
    values = corr(nLags:end) / denom;
    values = values(:);
end


function score = max_normalized_xcorr(x, template)
    x = double(x(:));
    template = double(template(:));
    x = x - mean(x);
    template = template - mean(template);
    denom = norm(x) * norm(template) + 1e-12;
    corr = xcorr(x, template, 'none') / denom;
    score = max(corr);
end


function score = cosine_similarity(x, template)
    x = double(x(:));
    template = double(template(:));
    score = dot(x, template) / (norm(x) * norm(template) + 1e-12);
end
