function summary = matlab_course_spectrum_demo(outputDir)
%MATLAB_COURSE_SPECTRUM_DEMO Plot in-class FFT spectra and score matrix.
%
% This demo uses the four case files:
%   get, boy, did, zero
% It creates:
%   1) one short-time FFT template spectrum for each consonant
%   2) a score matrix showing template-matching detection results

srcDir = fileparts(mfilename('fullpath'));
caseDir = fullfile(srcDir, '..', 'cases');

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(srcDir, '..', 'figures', 'course_matlab');
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

templatePaths = {
    fullfile(caseDir, 'case_g_get.wav')
    fullfile(caseDir, 'case_b_boy.wav')
    fullfile(caseDir, 'case_d_did.wav')
    fullfile(caseDir, 'case_z_zero.wav')
};
labels = {'g', 'b', 'd', 'z'};
words = {'get', 'boy', 'did', 'zero'};

[~, bank] = matlab_course_template_detector(templatePaths{1}, templatePaths, labels);

scoreMatrix = zeros(numel(words), numel(labels));
predLabels = strings(numel(words), 1);
for i = 1:numel(words)
    result = matlab_course_template_detector(templatePaths{i}, bank);
    scoreMatrix(i, :) = result.scores(:).';
    predLabels(i) = result.bestLabel;
end

figure('Color', 'w', 'Name', 'Course FFT spectra');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
for i = 1:numel(labels)
    nexttile;
    plot(bank.freq_hz, bank.templates(i, :), 'LineWidth', 1.4);
    xlim([0, 8000]);
    grid on;
    xlabel('Frequency (Hz)');
    ylabel('Normalized magnitude');
    title(sprintf('/%s/ in \"%s\"', labels{i}, words{i}));
end
exportgraphics(gcf, fullfile(outputDir, 'example_short_time_spectra.png'), 'Resolution', 180);

figure('Color', 'w', 'Name', 'Course detector scores');
imagesc(scoreMatrix);
axis equal tight;
colormap(parula);
colorbar;
xticks(1:numel(labels));
xticklabels(labels);
yticks(1:numel(words));
yticklabels(words);
xlabel('Template label');
ylabel('Query word');
title('Normalized correlation scores');
for r = 1:size(scoreMatrix, 1)
    for c = 1:size(scoreMatrix, 2)
        text(c, r, sprintf('%.2f', scoreMatrix(r, c)), ...
            'HorizontalAlignment', 'center', ...
            'Color', 'w', ...
            'FontSize', 10, ...
            'FontWeight', 'bold');
    end
end
exportgraphics(gcf, fullfile(outputDir, 'example_score_matrix.png'), 'Resolution', 180);

summary = struct();
summary.labels = string(labels(:));
summary.words = string(words(:));
summary.predLabels = predLabels;
summary.scoreMatrix = scoreMatrix;
summary.outputDir = string(outputDir);
summary.bank = bank;
end
