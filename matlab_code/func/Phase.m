classdef Phase < handle

      
   properties
      dt;
      z;
      zd;
      zdd;
   end
   
        
    
methods

    function obj = Phase(phase_dt)
        obj.dt = phase_dt;
        
        obj = obj.derivatives;

    end

   function obj = derivatives(obj)
    % Phase and derivatives
 
        dt = obj.dt;
        
        z   = dt:dt:1; 
        z = z';
        zD  = diff(z)./dt;
        zD  = [zD;  zD(end)];
        zDD = diff(zD)./dt;
        zDD = [zDD ; zDD(end)];

        obj.z   = z;
        obj.zd  = zD;
        obj.zdd = zDD;
   end
   
   
    
end %methods


end