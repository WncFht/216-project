function summary = matlab_case_demo(outputDir)
%MATLAB_CASE_DEMO Preview the four case clips with the strict course detector.
%
% The demo reads case_manifest.csv, runs matlab_course_template_detector
% on each clip, and plots the waveform together with the four label scores.

srcDir = fileparts(mfilename('fullpath'));
caseDir = fullfile(srcDir, '..', 'cases');
manifestPath = fullfile(caseDir, 'case_manifest.csv');

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(srcDir, '..', 'figures', 'course_matlab');
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

T = readtable(manifestPath, 'TextType', 'string');
labels = string(T.label);
words = string(T.word);
files = string(T.filename);

nCases = height(T);
scoreMat = zeros(nCases, 4);
predLabels = strings(nCases, 1);

figure('Color', 'w', 'Name', 'VE216 voiced consonant cases');
tiledlayout(nCases, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:nCases
    wavPath = fullfile(caseDir, char(files(i)));
    result = matlab_course_template_detector(wavPath);

    [x, sr] = audioread(wavPath);
    if size(x, 2) > 1
        x = mean(x, 2);
    end
    timeAxis = (0:numel(x)-1) / sr;

    nexttile;
    plot(timeAxis, x, 'k', 'LineWidth', 1.0);
    grid on;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('%s /%s/', char(words(i)), char(labels(i))));

    nexttile;
    b = bar(1:numel(result.scores), result.scores, 0.72);
    b.FaceColor = 'flat';
    b.CData = repmat([0.20 0.45 0.80], numel(result.scores), 1);
    [~, bestIdx] = max(result.scores);
    b.CData(bestIdx, :) = [0.85 0.33 0.10];
    ax = gca;
    xticks(1:numel(result.labels));
    xticklabels(cellstr(result.labels));
    ax.XTickLabelRotation = 0;
    ax.YGrid = 'on';
    ylabel('Score');
    ylim([0, max(1.02, max(result.scores) * 1.12)]);
    title(sprintf('Pred: %s', char(result.bestLabel)));

    predLabels(i) = result.bestLabel;
    scoreMat(i, :) = result.scores(:).';

    fprintf('%s (%s) -> predicted: %s | best=%.3f | scores=[%s]\n', ...
        char(T.case_id(i)), ...
        char(words(i)), ...
        char(result.bestLabel), ...
        result.bestScore, ...
        strjoin(arrayfun(@(k) sprintf('%s:%.3f', char(result.labels(k)), result.scores(k)), ...
            1:numel(result.scores), 'UniformOutput', false), ', '));
end

exportgraphics(gcf, fullfile(outputDir, 'case_overview.png'), 'Resolution', 180);

summary = struct();
summary.outputDir = string(outputDir);
summary.labels = labels;
summary.words = words;
summary.predLabels = predLabels;
summary.scoreMatrix = scoreMat;
summary.manifest = T;
end
