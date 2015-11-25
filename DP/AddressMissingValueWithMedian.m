function [s] = AddressMissingValueWithMedian(m)
% AddressMissingValueWithMedian - fill missing values(nan) with median by
%                                 respective feature
%     [s] = AddressMissingValueWithMedian(m)
%
%     name                      value
%     s    new matrix by filling out missing values with median
%
%     m    input matrix, it may contained some nan fields
%
% Hins Pan, updated on 2015.11.24
    [~, col] = size(m);
    s = m;
    for i = 2 : col
        [x, ~] = find(~isnan(m(:,i)));
        median = mode(m(x, i));
        [x, ~] = find(isnan(m(:,i)));
        s(x, i) = median;
    end
return