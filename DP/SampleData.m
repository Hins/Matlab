function [posDataSet, negDataSet, trainSet, validationSet, variance, stdDeviation] = SampleData(m, posLabel, negLabel, ratio)
% SampleData - sample data at random
%     [posDataSet, negDataSet, trainSet, validationSet, variance, stdDeviation] = SampleData(m, posLabel, negLabel, ratio)
%     sample data at random, however obtained pos/neg samples' ratio the
%     same both in trainSet and validationSet
%
%        name               value
%     posDataSet       positive data set
%
%     negDataSet       negtive data set
%
%     trainSet         training set
%
%     validationSet    validation set
%
%     m                raw data set
%
%     label            data label
%
%     ratio            sample ratio
%
% Hins Pan 2015.10.26

    tic;
    
    % Parameter check
    narginchk(4, Inf);
    if (~ismatrix(m))
        error(message('m should be a matrix!'));
    end
    
    if (~isnumeric(m))
        error(message('m contained non-real number!'));
    end
    
    if (~isnumeric(ratio))
        error(message('ratio should be a non-real number!'));
    end
    
    [~, col] = size(m);
    % Extract pos/neg sample;
    posSample = m(find(m(:,1) == posLabel), 1:col);
    negSample = m(find(m(:,1) == negLabel), 1:col);
    [posRow, ~] = size(posSample);
    [negRow, ~] = size(negSample);

    % Calculate out pos/neg random sample size
    posSampleSize = int64(posRow * ratio);
    negSampleSize = int64(negRow * ratio);
    
    % Generate pos/neg random sample set
    n = randperm(posRow, posSampleSize);
    posDataSet = posSample(n,1:col);

    n = randperm(negRow, negSampleSize);
    negDataSet = negSample(n,1:col);
    % Merge train set;
    trainSet = [posDataSet;negDataSet];
    validationSet = setdiff(m, trainSet, 'rows');
    
    variance = zeros(col - 1, 1);
    stdDeviation = zeros(col - 1, 1);
    for i = 2:col
        variance(i - 1, 1) = var(m(:,i));
        stdDeviation(i - 1, 1) = std(m(:,i));
    end
    
    toc;
end