function matlab_case_demo()
%MATLAB_CASE_DEMO Preview four case clips and run the detector.

srcDir = fileparts(mfilename('fullpath'));
caseDir = resolve_case_root(srcDir);
manifestPath = fullfile(caseDir, 'case_manifest.csv');
bankPath = fullfile(caseDir, 'course_template_bank.json');

T = load_case_manifest(manifestPath);

caseIdCol = resolve_table_column(T, {'case_id', 'caseid', 'id'});
labelCol = resolve_table_column(T, {'label', 'target_label', 'target'});
wordCol = resolve_table_column(T, {'word'});
fileCol = resolve_table_column(T, {'filename', 'file_name', 'wav_filename', 'wav_path', 'source_wav_path'});

nCases = height(T);
figure('Color', 'w', 'Name', 'VE216 voiced consonant cases');
tiledlayout(nCases, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:nCases
    fileValue = string(T.(fileCol)(i));
    wavPath = resolve_case_audio_path(caseDir, fileValue);
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

function T = load_case_manifest(manifestPath)
rawHeaders = read_manifest_headers(manifestPath);
T = table();

try
    opts = detectImportOptions(manifestPath, 'TextType', 'string');
    try
        opts.VariableNamingRule = 'preserve';
    catch
        % Older MATLAB releases do not expose this property.
    end
    T = readtable(manifestPath, opts);
catch
    try
        T = readtable(manifestPath, 'TextType', 'string');
    catch
        T = table();
    end
end

if isempty(T) || width(T) ~= numel(rawHeaders) || has_auto_varnames(T.Properties.VariableNames)
    T = load_case_manifest_textscan(manifestPath, rawHeaders);
    return;
end

if ~isempty(rawHeaders) && numel(rawHeaders) == width(T)
    try
        T.Properties.VariableNames = rawHeaders;
    catch
        % If the table keeps imported names, the downstream alias lookup still works.
    end
end
end

function T = load_case_manifest_textscan(manifestPath, rawHeaders)
fid = fopen(manifestPath, 'r');
if fid < 0
    error('Unable to open case_manifest.csv: %s', manifestPath);
end
cleanup = onCleanup(@() fclose(fid));

formatSpec = repmat('%q', 1, numel(rawHeaders));
data = textscan( ...
    fid, ...
    formatSpec, ...
    'Delimiter', ',', ...
    'HeaderLines', 1, ...
    'ReturnOnError', false, ...
    'EndOfLine', '\n');

validNames = matlab.lang.makeValidName(rawHeaders, 'ReplacementStyle', 'delete');
columns = cell(1, numel(validNames));
for i = 1:numel(validNames)
    columns{i} = string(data{i});
end
T = table(columns{:}, 'VariableNames', validNames);
end

function headers = read_manifest_headers(manifestPath)
fid = fopen(manifestPath, 'r');
if fid < 0
    error('Unable to open case_manifest.csv: %s', manifestPath);
end
cleanup = onCleanup(@() fclose(fid));
headerLine = fgetl(fid);
if ~ischar(headerLine)
    error('case_manifest.csv is empty: %s', manifestPath);
end

parts = strsplit(headerLine, ',');
headers = cell(1, numel(parts));
for i = 1:numel(parts)
    headers{i} = strip_bom_and_trim(parts{i});
end
end

function out = strip_bom_and_trim(textValue)
out = char(textValue);
if ~isempty(out) && out(1) == char(65279)
    out(1) = [];
end
out = strtrim(out);
end

function tf = has_auto_varnames(names)
tf = false;
for i = 1:numel(names)
    if ~isempty(regexp(names{i}, '^Var\d+$', 'once'))
        tf = true;
        return;
    end
end
end

function wavPath = resolve_case_audio_path(caseDir, fileValue)
fileValue = strtrim(char(fileValue));
if isfile(fileValue)
    wavPath = fileValue;
    return;
end

candidateDirs = {
    caseDir
    fileparts(caseDir)
    fullfile(caseDir, 'cases')
    fullfile(fileparts(caseDir), 'cases')
};
for i = 1:numel(candidateDirs)
    candidate = fullfile(candidateDirs{i}, fileValue);
    if isfile(candidate)
        wavPath = candidate;
        return;
    end
end

error('Unable to locate audio file: %s', fileValue);
end

function caseRoot = resolve_case_root(startDir)
candidateDirs = {
    fullfile(startDir, 'cases')
    startDir
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
vars = normalize_names(T.Properties.VariableNames);
for i = 1:numel(wantedNames)
    wanted = lower(strtrim(char(wantedNames{i})));
    idx = find(strcmp(vars, wanted), 1);
    if isempty(idx)
        idx = find(contains(vars, wanted), 1);
    end
    if ~isempty(idx)
        colName = T.Properties.VariableNames{idx};
        return;
    end
end
error( ...
    'Missing required column in case_manifest.csv: %s. Available columns: %s', ...
    strjoin(string(wantedNames), ', '), ...
    strjoin(string(T.Properties.VariableNames), ', '));
end

function names = normalize_names(rawNames)
names = cell(1, numel(rawNames));
for i = 1:numel(rawNames)
    names{i} = lower(strtrim(strip_bom_and_trim(rawNames{i})));
end
end
