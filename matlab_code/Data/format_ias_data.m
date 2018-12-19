function data_ias = format_ias_data(data)
% This is simply to put the data on the format that is used by the IAS
% Class System.

    nDemo = size(data(1).q,1);
    for k=1:nDemo
        data_ias{k,:} = [data(1).q(k,:)' data(2).q(k,:)'...
                         data(3).q(k,:)' data(4).q(k,:)'];
    end

end