function [aucs, models, avgPrecisions, maxPrecisions, avgRecalls, maxRecalls] = LRByDiffRatio(m, posLabel, negLabel, labelNormalization, slideMethod)
% LRByDiffRatio - Logistic regression by different ratio
%     [aucs, models, avgPrecisions, maxPrecisions, avgRecalls, maxRecalls] = LRByDiffRatio(m, posLabel, negLabel, labelNormalization, slideMethod)
%
%     name    value
%     aucs                  AUC value out of different ratio
%
%     models                Model out of different ratio, each model is a vector
%
%     avgPrecision          average precision value
%
%     maxPrecision          maximum precision value
%
%     avgRecall             average recall value
%
%     maxRecall             maximum recall value
%
%     m                     raw matrix including label and unaddressed-features
%
%     posLabel              positive samples' label
%
%     negLabel              negative samples' label
%
%     labelNormalization    1 meant we should do label normalization,
%                           that is we should replace non-zero negative labels into 0
%
%     slideMethod           0 meant Gaussian normalization, 1 meant MinMax normalization
%
% Hins Pan 2015.11.6
    % Define various ratio allocation by hard-code, maybe we could fix it
    % by a parameter.
    ratioMatrix = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
    
    % Parameter check
    narginchk(5, Inf);
    if (~ismatrix(m))
        error(message('m should be a matrix!'));
    end
    
    if (~isnumeric(m))
        error(message('m contained non-real numbers!'));
    end
    [~, col] = size(ratioMatrix);
    aucs = zeros(1, col);
    avgPrecisions = zeros(1, col);
    maxPrecisions = zeros(1, col);
    avgRecalls = zeros(1, col);
    maxRecalls = zeros(1, col);
    [~, mcol] = size(m);
    models = zeros(mcol, col);
    
    for i = 1:col
        % Start timing
        tic;
        % Sample data
        [~,~,t,v] = SampleData(m, posLabel, negLabel, ratioMatrix(1,i));
        disp(strcat(int2str(i),' Sample complete'));
        % Logistic regression
        [~,~,~,model,auc,~,~,avgPrecision,maxPrecision,avgRecall,maxRecall] = LogisticRegression(t, v, labelNormalization, slideMethod);
        disp('LR training complete');
        aucs(1, i) = auc;
        avgPrecisions(i, 1) = avgPrecision;
        maxPrecisions(i, 1) = maxPrecision;
        avgRecalls(i, 1) = avgRecall;
        maxRecalls(i, 1) = maxRecall;
        models(:, i) = model;
        toc;
    end
end