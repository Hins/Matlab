function [norm_m] = MinMaxNormalization(m)
% MinMaxNormalization - Min-Max normalization: value = (value - min) / (max - min)
%     [norm_m] = MinMaxNormalization(m)
%
%       name          value
%     norm_m    normalized matrix
%
%     m         raw matrix
%
% Hins Pan 2015.10.23

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
    for i = 1:col
        minValue = min(m(:,i));
        maxValue = max(m(:,i));
        if minValue ~= maxValue
            for j = 1:row
                norm_m(j, i) = (m(j, i) - minValue) / (maxValue - minValue);
            end
        end
    end
end