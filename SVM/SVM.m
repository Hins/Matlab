function [svm_struct, auc, x, y, avgPrecision, maxPrecision, avgRecall, maxRecall] = SVM(trainSet, validationSet, labelNormalization, slideMethod, options, method, kernelFunction)
% SVM - Train a SVM model
%     [svm_struct, auc, x, y, avgPrecision, maxPrecision, avgRecall, maxRecall] = SVM(trainSet, validationSet, labelNormalization, slideMethod, options, method, kernelFunction)
%     Train a svm model, return svm model and auc/precision/recall metrics
%
%       Name                                  Value
%     'trainSet'              training set, the schema is "<label> <f1> <f2> ... <fn>",
%                             one sample one line
%     
%     'validationSet'         validation set, the schema is the same with trainSet
%
%     'labelNormalization'    accepts arbitary input, 1 meant we should
%                             replace -1 with 0 against label column to adapter train/classify
%                             procedures.
%
%     'slideMethod'           accepts arbitary input, 0 meant Gaussian normalization, 1 meant
%                             min-max normalization.
%
%     'options'               Options structure created using either STATSET or
%                             OPTIMSET.
%                             * When you set 'method' to 'SMO' (default),
%                             create the options structure using STATSET.
%                             Applicable options:
%                             'Display'  Level of display output.  Choices
%                                     are 'off' (the default), 'iter', and
%                                     'final'. Value 'iter' reports every
%                                     500 iterations.
%                             'MaxIter'  A positive integer specifying the
%                                     maximum number of iterations allowed.
%                                     Default is 15000 for method 'SMO'. If
%                                     you counldn't estimate iteration
%                                     scale, please set it into Inf
%                             * When you set method to 'QP', create the
%                             options structure using OPTIMSET. For details
%                             of applicable options choices, see QUADPROG
%                             options. SVM uses a convex quadratic program,
%                             so you can choose the 'interior-point-convex'
%                             algorithm in QUADPROG.
%
%     'method'                A string specifying the method used to find the
%                             separating hyperplane. Choices are:
%                             'SMO' - Sequential Minimal Optimization (SMO)
%                                  method (default). It implements the L1
%                                  soft-margin SVM classifier.
%                             'QP'  - Quadratic programming (requires an
%                                  Optimization Toolbox license). It
%                                  implements the L2 soft-margin SVM
%                                  classifier. Method 'QP' doesn't scale
%                                  well for TRAINING with large number of
%                                  observations.
%                             'LS'  - Least-squares method. It implements the
%                                  L2 soft-margin SVM classifier.
%
%     'kernelFunction'        A string or a function handle specifying the
%                             kernel function used to represent the dot
%                             product in a new space. The value can be one of
%                             the following:
%                             'linear'     - Linear kernel or dot product
%                                         (default). In this case, svmtrain
%                                         finds the optimal separating plane
%                                         in the original space.
%                             'quadratic'  - Quadratic kernel
%                             'polynomial' - Polynomial kernel with default
%                                         order 3. To specify another order,
%                                         use the 'polyorder' argument.
%                             'rbf'        - Gaussian Radial Basis Function
%                                         with default scaling factor 1. To
%                                         specify another scaling factor,
%                                         use the 'rbf_sigma' argument.
%                             'mlp'        - Multilayer Perceptron kernel (MLP)
%                                         with default weight 1 and default
%                                         bias -1. To specify another weight
%                                         or bias, use the 'mlp_params'
%                                         argument.
%                             function     - A kernel function specified using
%                                         @(for example @KFUN), or an
%                                         anonymous function. A kernel
%                                         function must be of the form
% 
%                                         function K = KFUN(U, V)
% 
%                                         The returned value, K, is a matrix
%                                         of size M-by-N, where M and N are
%                                         the number of rows in U and V
%                                         respectively.
%
%     'svm_struct'            svm model
%
%     'auc'                   AUC metric
%
%     'x'                     ROC curve's x-axises
%
%     'y'                     ROC curve's y-axises
%
%     'avgPrecision'          average precision
%
%     'maxPrecision'          maximum precision
%
%     'avgRecall'             average recall
%
%     'maxRecall'             maximum recall
%
% Hins Pan 2015.11.20

    tic;

    % Parameter check
    narginchk(7, Inf);
    [~, tCol] = size(trainSet);
    [~, vCol] = size(validationSet);
    if tCol ~= vCol
        error(message('Column size is different between training set and validation set'));
    end
    
    if ~isnumeric(trainSet) || ~isnumeric(validationSet)
        error(message('TrainSet or validationSet contained non-real numbers'));
    end
    
    if ~isstruct(options)
        error(message('options should be a struct'));
    end
    
    if ~ischar(method)
        error(message('method should be a string'));
    end
    
    if ~ischar(kernelFunction)
        error(message('kernelFunction should be a string'));
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
    
    svm_struct = svmtrain(trainData, trainLabel, 'options', options, 'method', method, 'Showplot', true, 'kernel_function', kernelFunction);
    svm_result = svmclassify(svm_struct, validationData, 'Showplot', true);
    [auc, x, y, avgPrecision, maxPrecision, avgRecall, maxRecall] = plot_roc(svm_result, validationLabel);
    
    toc;
end