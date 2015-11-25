% Logistic regression
% Return value: validationData(validation data without label),
%               validationLabel(validation label),
%               predication(predication result),
%               LRModel(LR model), 
%               auc(auc metric), 
%               X(x-coordinate in ROC graph), 
%               Y(y-coordinate in ROC graph),
%               avgPrecision(average precision),
%               maxPrecision(maximum precision),
%               avgRecall(average recall),
%               maxRecall(maximum recall)
% Parameter: trainSet(schema is that first column is label, others are
% features), 
%            validationSet(schema is the same with trainSet), 
%            labelNormalization(1 will address minus value into 0 to fit
%            glmfit),
%            labelNormalization(0 is Gaussian distribution, 1 is min-max
%            distribution);
% Hins Pan, 2015.10.23
function [validationData, validationLabel, predication, LRModel, auc, X, Y, avgPrecision, maxPrecision, avgRecall, maxRecall] = LogisticRegression(trainSet, validationSet, labelNormalization, slideMethod)
    [tRow, tCol] = size(trainSet);
    [~, vCol] = size(validationSet);
    if tCol ~= vCol
        return
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
    LRModel = glmfit(trainData, [trainLabel ones(tRow,1)], 'binomial', 'link', 'logit');
    disp('Training complete');
    % Logistic regression predication;
    predication = glmval(LRModel, validationData, 'logit');
    disp('Predication complete');
    % Calculate ROC/AUC metrics;
    [auc, X, Y, avgPrecision, maxPrecision, avgRecall, maxRecall] = plot_roc(predication, validationLabel);
    disp('Metric computation complete');
end