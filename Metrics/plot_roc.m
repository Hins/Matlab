function [auc,x,y,avgPrecision,maxPrecision,avgRecall,maxRecall] = plot_roc(predict, ground_truth)
% plot_roc - calculate some classification metrics, as well as AUC, precision and recall
%     [auc,x,y,avgPrecision,maxPrecision,avgRecall,maxRecall] = plot_roc(predict, ground_truth)
%     calculate auc/precision/recall, meanwhile draw the ROC curve
%
%        name            value
%     auc             auc metric
%
%     x               x-axis in ROC curve
%
%     y               y-axis in ROC curve
%
%     avgPrecision    average precision value
%
%     maxPrecision    maximum precision value
%
%     avgRecall       average recall value
%
%     maxRecall       maximum recall value
%
%     predict         predication result by model
%
%     ground_truth    real label value
%
% Hins Pan, updated on 2015.11.24

    tic;
    
    narginchk(2, Inf);
    if (~isvector(predict) || ~isvector(ground_truth))
        error(message('predict or ground_truth is not a vector!'));
    end
    
    if (size(predict, 1) ~= size(ground_truth, 1))
        error(message('row size is inconsistent between ground_truth and predict!'));
    end
    
    if (~isnumeric(predict) || ~isnumeric(ground_truth))
        error(message('predict or ground_truth contained non-real numbers!'));
    end
    %Parameter check
    avgPrecision = 0.0;
    maxPrecision = 0.0;
    avgRecall = 0.0;
    maxRecall = 0.0;
   
    pos_num = sum(ground_truth==1);
    neg_num = sum(ground_truth==0);

    m = size(ground_truth,1);
    [~,Index]=sort(predict);
    ground_truth=ground_truth(Index);
    P = size(find(ground_truth == 1));

    x=zeros(m+1,1);
    y=zeros(m+1,1);

    auc=0;
    x(1)=1;y(1)=1;

    for i=2:m
        TP = sum(ground_truth(i:m)==1);
        FP = sum(ground_truth(i:m)==0);
        x(i) = FP/neg_num;
        y(i) = TP/pos_num;
        auc = auc+(y(i)+y(i-1))*(x(i-1)-x(i))/2;

        if maxRecall < TP(1,1)/P(1,1)
            maxRecall = TP(1,1)/P(1,1);
        end
        if maxPrecision < TP(1,1) / (TP(1,1) + FP(1,1))
            maxPrecision = TP(1,1) / (TP(1,1) + FP(1,1));
        end

        avgRecall = avgRecall + TP(1,1)/P(1,1);
        avgPrecision = avgPrecision + TP(1,1) / (TP(1,1) + FP(1,1));
    end;

    x(m+1)=0;
    y(m+1)=0;
    auc=auc+y(m)*x(m)/2;
    avgRecall = avgRecall / m;
    avgPrecision = avgPrecision / m;
    plot(x,y);
    
    toc;
end