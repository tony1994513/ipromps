classdef Observation < handle

      
   properties
      set;
      joint;
      jointvel;
      index;
      stdev;
      q;
      qdot;
   end
   
        
    
methods

    function obj = Observation(set, stdev, param)
        obj.set   = set;
        obj.joint = param.observedJointPos;
        obj.jointvel = param.observedJointVel;
        obj.stdev = stdev;
        
        % prepare the vector of observations by filling it with zeros
        for j = 1:param.nTotalJoints
            obj.q(j,:) = 0.*ones(1, param.nTraj);
            obj.qdot(j,:) = 0.*ones(1, param.nTraj);
        end
        
    end

    function obj = measuredIndexes(obj, index, data)
        obj.index = index;

        for j = obj.joint
            obj.q(j,index)    = data(j).q(obj.set, index);            
        end
        for j = obj.jointvel
            obj.qdot(j,index) = data(j).qdot(obj.set, index);
        end

    end

    
end %methods


end


%     clear obs
%     obs(1).parent.set    = 2;
%     obs(1).parent.joints = [1 2];
%     obs(1).parent.time   = [   [50:55]   [65]   80   90    [10:15]];
%     obs(1).parent.stdev  = 0.01;
%     for j = obs(1).parent.joints
%         index_observed_data = obs(1).parent.time;
%         data_set = obs(1).parent.set;
%         obs(j).q = NaN.*ones(1,data(1).nSize) ;
%         obs(j).q(index_observed_data ) = data(j).q(data_set, index_observed_data );
%     end
