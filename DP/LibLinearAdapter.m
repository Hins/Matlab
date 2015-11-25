function [str] = LibLinearAdapter(m)
% LibLinearAdapter - Transform data format into LibLinear schema
%   Return value: str(string content)
%
%   Parameters: m(data matrix)
%
% HinsPan 2015.11.19
    [row, col] = size(m);
    str = '';
    for i = 1 : row   
        subStr = '';
        for j = 1:col
            % Address label specially;
            if j == 1
                if m(i, j) == 1
                    subStr = strcat(subStr, '+1');
                else
                    subStr = strcat(subStr, int2str(m(i, j)));
                end
            else
                % Add sequence number and blank among features;
                subStr = strcat(subStr, [' ', strcat(strcat(int2str(j - 1), ':'), num2str(m(i, j)))]);
            end
        end
        str = strcat(str, strcat(subStr, '\n'));
    end
end