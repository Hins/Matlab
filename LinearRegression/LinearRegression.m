function [LRModel, auc, X, Y, avgPrecision, maxPrecision, avgRecall, maxRecall] = LinearRegression(trainSet, validationSet, labelNormalization, slideMethod)
% Linear regression - train a linear regression model
%     [validationData, validationLabel, predication, LRModel, auc, X, Y, avgPrecision, maxPrecision, avgRecall, maxRecall] = LinearRegression(trainSet, validationSet, labelNormalization, slideMethod)
%     Train a linear regression model, return lr model and
%     auc/precision/recall metrics.
%
%        name                                    value
%     'trainSet'              training set, schema is that first column is label, others are features
%
%     'validationSet'         validation set, schema is the same with trainSet
%
%     'labelNormalization'    1 will address minus value into 0 to fit glmfit
%
%     'labelNormalization'    0 is Gaussian distribution, 1 is min-max distribution
%     
%     'LRModel'               Linear regression model
%
%     'auc'                   auc metric
%
%     'X'                     x-coordinate in ROC curve
%
%     'Y'                     y-coordinate in ROC curve
%
%     'avgPrecision'          average precision
%
%     'maxPrecision'          maximum precision
%
%     'avgRecall'             average recall
%
%     'maxRecall'             maximum recall
%
% Hins Pan, 2015.10.23

    tic;
    
    [~, tCol] = size(trainSet);
    [~, vCol] = size(validationSet);
    if tCol ~= vCol
        error(message('Column size is different between training set and validation set'));
    end
    if ~isnumeric(trainSet) || ~isnumeric(validationSet)
        error(message('TrainSet or validationSet contained non-real numbers'));
    end
    
    trainLabel = trainSet(:,1);
    trainData = trainSet(:,2:tCol);
    validationLabel = validationSet(:,1);
    validationData = validationSet(:,2:tCol);
    % Address minus value into 0 to fit glmfit model;
    if labelNormalization == 1
        index = trainLabel == -1;
        trainLabel(index) = 0;
        index = validationLabel == -1;
        validationLabel(index) = 0;
    end
    
    disp('Lable normalization complete');
    
    % Gaussian normalization
    if slideMethod == 0
        trainData = GaussianNormalization(trainData);
        validationData = GaussianNormalization(validationData);
    % Max-min normalization
    elseif slideMethod == 1
        trainData = MinMaxNormalization(trainData);
        validationData = MinMaxNormalization(validationData);
    end
    
    disp('Data normalization complete');
    
    % Logistic regression training;
    %LRModel = glmfit(trainData, [trainLabel ones(tRow,1)], 'binomial', 'link', 'logit');
    LRModel = glmfit(trainData, trainLabel, 'normal', 'link', 'identity');
    disp('Training complete');
    % Linear regression predication;
    predication = glmval(LRModel, validationData, 'identity');
    disp('Predication complete');
    % Calculate ROC/AUC metrics;
    
    [auc, X, Y, avgPrecision, maxPrecision, avgRecall, maxRecall] = plot_roc(predication, validationLabel);
    disp('Metric computation complete');
    
    toc;
end