function [aucs, models, avgPrecisions, maxPrecisions, avgRecalls, maxRecalls] = LassoByDiffRatio(m, posLabel, negLabel, alpha, lambda, numLambda, CV, labelNormalization, slideMethod)
% LassoByDiffRatio - LR with lasso/ridge/elasticNet regularization by different ratio
%     [aucs, models, avgPrecisions, maxPrecisions, avgRecalls, maxRecalls] = LassoByDiffRatio(m, posLabel, negLabel, alpha, lambda, numLambda, CV, labelNormalization, slideMethod)
% 
%        name                             value
%     aucs                  AUC value out of different ratio)
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
%     alpha                 parameter to control lasso/ridge/elasticNet degree, 1 meant lasso, 0 meant ridge, (0,1) meant elasticNet)
%
%     lambda                learning rate of regularization items
%
%     numLambda             how many learning rates we should use
%
%     CV                    cross-validation size
%
%     labelNormalization    1 meant we should do label normalization,
%                           that is we should replace non-zero negative labels into 0)
%
%     slideMethod           0 meant Gaussian normalization, 1 meant MinMax
%                           normalization
%
% Hins Pan, 2015.11.6
    % Define various ratio allocation by hard-code, maybe we could fix it
    % by a parameter.
    ratioMatrix = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
    
    % Parameter check
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
        tic;
        % Sample data
        [~,~,t,v] = SampleData(m, posLabel, negLabel, ratioMatrix(1,i));
        disp(strcat(int2str(i),' Sample complete'));
        % Lasso regression
        [auc, model, avgPrecision, maxPrecision, avgRecall, maxRecall] = Lasso(t(:,2:mcol), t(:,1), v(:,2:mcol), v(:,1), alpha, lambda, numLambda, CV, labelNormalization, slideMethod);
        disp('Lasso training complete');
        aucs(1, i) = auc;
        models(:, i) = model;
        avgPrecisions(i, 1) = avgPrecision;
        maxPrecisions(i, 1) = maxPrecision;
        avgRecalls(i, 1) = avgRecall;
        maxRecalls(i, 1) = maxRecall;
        toc;
    end
    
    toc;
end