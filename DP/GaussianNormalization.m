function [norm_m] = GaussianNormalization(m)
% Gaussian normalization: suppose current data followed Gaussian distribution
%                         value = (value - mean) / std_variance
%     [norm_m] = GaussianNormalization(m)
%
%       name          value
%     norm_m    normalizat matrix
%
%     m         raw matrix
%
% Hins Pan, 2015.10.23
    
    %Parameter check
    narginchk(1, Inf);
    if (~ismatrix(m))
        error(message('m should be a matrix!'));
    end
    
    if (~isnumeric(m))
        error(message('m contained non-real number!'));
    end
    
    [row, col] = size(m);
    norm_m = zeros(row, col);
    for i=1:col
        avg = mean(m(:,i));
        std_variance = std(m(:,i));
        for j=1:row
            norm_m(j, i) = (m(j, i) - avg) / std_variance;
        end
    end
end