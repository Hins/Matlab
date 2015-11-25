function [auc, model, allModels, avgPrecision, maxPrecision, avgRecall, maxRecall] = Lasso(t, lt, v, lv, alpha, lambda, numLambda, CV, labelNormalization, slideMethod)
% Lasso - train a logistic model by L1/L2 regularizations
%     [auc, model, allModels, avgPrecision, maxPrecision, avgRecall, maxRecall] = Lasso(t, lt, v, lv, maxRecall, lambda, numLambda, CV, labelNormalization, slideMethod)
%     L1 regularization is used to filter some sparse features, L2 is rigde
%     regression, elasticNet is between L1 and L2.
%     Differentiate with ordinal logistic regression, with regularizations
%     Lasso will get better result by fitlering better features and removing overfitting.
%     
%           name                     value
%     'auc'                   auc metrics by ROC curve
%
%     'model'                 logistic regression model
%
%     'allModels'             we will get multiple models by different learning
%                             rate, thus we return them together.
%
%     'avgPrecision'          average precision value
%
%     'maxPrecision'          maximum precision value
%
%     'avgRecall'             average recall value
%
%     'maxRecall'             maximum recall value
%
%     't'                     training set matrix
%
%     'lt'                    training set's label vector
%
%     'v'                     validation set matrix
%
%     'lv'                    validation set's label vector
%
%     'alpha'                 Elastic net mixing value, or the relative balance
%                             between L2 and L1 penalty (default 1, range (0,1]).
%                             Alpha=1 ==> lasso, otherwise elastic net.
%                             Alpha near zero ==> nearly ridge regression.
%
%     'lambda'                Lambda values. Will be returned in return argument
%                             STATS in ascending order. The default is to have
%                             lassoglm generate a sequence of lambda values, based 
%                             on 'NumLambda' and 'LambdaRatio'. lassoglm will generate 
%                             a sequence, based on the values in X and Y, such that 
%                             the largest Lambda value is estimated to be just 
%                             sufficient to produce all zero coefficients B. 
%                             You may supply a vector of real, non-negative values 
%                             of lambda for lassoglm to use, in place of its default
%                             sequence.  If you supply a value for 'Lambda', 
%                             'NumLambda' and 'LambdaRatio' are ignored.
%
%     'numLambda'             The number of lambda values to use, if the parameter
%                             'Lambda' is not supplied (default 100). Ignored
%                             if 'Lambda' is supplied. lassoglm may return fewer
%                             fits than specified by 'NumLambda' if the deviance
%                             of the fits drops below a threshold percentage of
%                             the null deviance (deviance of the fit without
%                             any predictors X).
%
%     'CV'                    If present, indicates the method used to compute Deviance.
%                             When 'CV' is a positive integer K, lassoglm uses K-fold
%                             cross-validation.  Set 'CV' to a cross-validation 
%                             partition, created using CVPARTITION, to use other
%                             forms of cross-validation. You cannot use a
%                             'Leaveout' partition with lassoglm.                
%                             When 'CV' is 'resubstitution', lassoglm uses X and Y 
%                             both to fit the model and to estimate the deviance 
%                             of the fitted model, without cross-validation.  
%                             The default is 'resubstitution'.
%
%     'labelNormalization'    accepts arbitary input, 1 meant we should
%                             replace -1 with 0 against label column to adapter train/classify
%                             procedures.
%
%     'slideMethod'           accepts arbitary input, 0 meant Gaussian normalization, 1 meant
%                             min-max normalization.
%
% Hins Pan, updated on 2015.11.24

    tic;
    
    % Parameter check
    if (~ismatrix(t) || ~ismatrix(v))
        error(message('training/validation sets should be matrixes!'));
    end
    
    if (~isnumeric(t) || ~isnumeric(v))
        error(message('TrainSet or validationSet contained non-real numbers!'));
    end
    
    if (size(t,2) ~= size(v, 2))
        error(message('Schema is inconsistent between trainSet and validationSet!'));
    end
    
    if (~isvector(lt) || ~isvector(lv))
        error(messaget('TrainSet/validationSet label should be vectors!'));
    end
    
    if (~isnumeric(lt) || ~isnumeric(lv))
        error(message('TrainSet or validationSet label vectors contained non-real numbers!'));
    end
    
    if (~isnumeric(alpha))
        error(message('Parameter alpha should be a real number!'));
    end
    
    t_back = t;
    lt_back = lt;
    v_back = v;
    lv_back = lv;
    if labelNormalization == 1
        index = lt_back == -1;
        lt_back(index) = 0;
        index = lv_back == -1;
        lv_back(index) = 0;
    end
    
    disp('Lable normalization complete');
    
    % Gaussian normalization
    if slideMethod == 0
        t_back = GaussianNormalization(t_back);
        v_back = GaussianNormalization(v_back);
    % Max-min normalization
    elseif slideMethod == 1
        t_back = MinMaxNormalization(t_back);
        v_back = MinMaxNormalization(v_back);
    end
    
    disp('Data normalization complete');
    
    if CV ~= 0
        if lambda ~= 0
            [w, ~] = lassoglm(t_back, lt_back, 'binomial', 'Alpha', alpha, 'Lambda', lambda, 'CV', CV);
        else
            [w, ~] = lassoglm(t_back, lt_back, 'binomial', 'Alpha', alpha, 'NumLambda', numLambda, 'CV', CV);
        end
    else
        if lambda ~= 0
            [w, ~] = lassoglm(t_back, lt_back, 'binomial', 'Alpha', alpha, 'Lambda', lambda);
        else
            [w, ~] = lassoglm(t_back, lt_back, 'binomial', 'Alpha', alpha, 'NumLambda', numLambda);
        end
    end
    allModels = w;
    disp('Training complete');
    
    lassoFilledVec = zeros(1,1);
    if lambda ~= 0
        lassoModel = [lassoFilledVec; w];
    else
        if numLambda > 6
            lassoModel = [lassoFilledVec; w(:, 4)];
        else
            lassoModel = [lassoFilledVec; w(:, 2)];
        end
    end
    
    disp('Model construction complete');
    
    predication = glmval(lassoModel, v_back, 'logit');
    [auc, ~, ~, avgPrecision, maxPrecision, avgRecall, maxRecall] = plot_roc(predication, lv_back);
    
    disp('Predication complete'); 
    
    model = lassoModel;
    
    toc;
end