function matlab_case_demo()
%MATLAB_CASE_DEMO Preview the four case clips and run the detector.

srcDir = fileparts(mfilename('fullpath'));
caseDir = resolve_case_root(srcDir);
manifestPath = fullfile(caseDir, 'case_manifest.csv');
bankPath = fullfile(caseDir, 'course_template_bank.json');

T = readtable(manifestPath, 'TextType', 'string');
caseIdCol = resolve_table_column(T, {'case_id', 'caseid', 'id'});
labelCol = resolve_table_column(T, {'label', 'target_label', 'target'});
wordCol = resolve_table_column(T, {'word'});
fileCol = resolve_table_column(T, {'filename', 'file_name', 'wav_filename', 'wav_path', 'source_wav_path'});

figure('Color', 'w', 'Name', 'VE216 voiced consonant cases');
tiledlayout(height(T), 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:height(T)
    fileValue = string(T.(fileCol)(i));
    if isfile(fileValue)
        wavPath = char(fileValue);
    else
        wavPath = fullfile(caseDir, char(fileValue));
    end
    result = matlab_letter_detector(wavPath, bankPath);

    [x, sr] = audioread(wavPath);
    if size(x, 2) > 1
        x = mean(x, 2);
    end
    timeAxis = (0:numel(x)-1) / sr;

    nexttile;
    plot(timeAxis, x, 'k');
    grid on;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('%s /%s/', char(string(T.(wordCol)(i))), char(string(T.(labelCol)(i)))));

    nexttile;
    imagesc(result.features.logmel.');
    axis xy;
    colormap hot;
    colorbar;
    xlabel('Frame');
    ylabel('Mel bin');
    title(sprintf('Pred: %s', char(strjoin(result.predLabels, ', '))));

    fprintf('%s (%s) -> predicted: %s | best=%s (%.3f)\n', ...
        char(string(T.(caseIdCol)(i))), ...
        char(string(T.(wordCol)(i))), ...
        char(strjoin(result.predLabels, ', ')), ...
        char(result.bestLabel), ...
        result.bestScore);
end
end

function caseRoot = resolve_case_root(startDir)
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
    manifest = fullfile(candidateDirs{i}, 'case_manifest.csv');
    bank = fullfile(candidateDirs{i}, 'course_template_bank.json');
    if isfile(manifest) && isfile(bank)
        caseRoot = candidateDirs{i};
        return;
    end
end
error('Unable to locate the case folder. Expected case_manifest.csv and course_template_bank.json in the same folder, a cases/ subfolder, or the topic2_speech/cases layout.');
end

function colName = resolve_table_column(T, wantedNames)
vars = string(T.Properties.VariableNames);
lowerVars = lower(vars);
for i = 1:numel(wantedNames)
    wanted = lower(string(wantedNames{i}));
    idx = find(lowerVars == wanted, 1);
    if isempty(idx)
        idx = find(contains(lowerVars, wanted), 1);
    end
    if ~isempty(idx)
        colName = T.Properties.VariableNames{idx};
        return;
    end
end
error('Missing required column in case_manifest.csv: %s', strjoin(string(wantedNames), ', '));
end
