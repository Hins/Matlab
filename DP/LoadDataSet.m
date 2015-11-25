function [m, missFeatureVec] = LoadDataSet(path, row, col, termSeparator, lineSeparator)
% LoadDataSet - load data from file system, deserialize them into a special
% format
%     [m, missFeatureVec] = LoadDataSet(path, row, col, termSeparator, lineSeparator)
%     Load data and deserialize them into a special format.
%
%          name                          value
%     'path'              data path, which should represent a matrix
%
%     'row'               matrix's row
%
%     'col'               matrix's column
%
%     'termSeparator'     separator among terms
%
%     'lineSeparator'     separator among lines
%
%     'm'                 matrix
%
%     'missFeatureVec'    a vector, whose element meant miss value's
%     percentage
% 
% Hins Pan 2015.11.5

    tic;
    
    % Parameter check;
    if exist(path, 'file') == 0
        error(message('Sorry, the path is invalid!'));
    end
    if fix(row) ~= row || fix(col) ~= col
        error(message('row and col should be integer!'));
    end
    
    fid = fopen(path);
    i = 1;
    m = zeros(row, col + 1);
    missFeatureVec = zeros(col, 1);
    while ~feof(fid)
        line = fgetl(fid);
        S = regexp(line, termSeparator, 'split');
        S2 = regexp(S(2), lineSeparator, 'split');
        m(i, 1) = str2double(S{1});
        
        for j = 1 : col
            m(i, j + 1) = str2double(S2{1}(1, j));
        end
        
        i = i + 1;
    end
    
    for i = 2 : col + 1
        [r, ~] = size(find(isnan(m(:,i))));
        missFeatureVec(i - 1, 1) = r / row;
    end
    
    toc;
end