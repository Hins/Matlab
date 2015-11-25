function [trainSet, validationSet] = SampleDataRandom(m, ratio)
% SampleDataRandom - sample data at random
%     [trainSet, validationSet] = SampleDataRandom(m, ratio)
%     Sample data at random
%
%         name                value
%     'm'                raw data set, a matrix
%
%     'ratio'            sample ratio
%
%     'trainSet'         train set
%
%     'validationSet'    validation set
%
% Hins Pan 2015.10.26

    tic;
    
    % Parameter check;
    if ~isnumeric(ratio)
        error(message('ratio should be a real number.'));
    end
    if ratio < 0.0 || ratio > 1.0
        error(message('ratio should be less than 1 and more than 0'));
    end

    [row, col] = size(m);
    % Calculate out pos/neg random sample size
    size2 = int64(row * ratio);

    % Generate pos/neg random sample set
    n = randperm(row, size2);
    trainSet = m(n, 1:col);
    validationSet = m;
    validationSet(n,:) = [];
    
    toc;
end