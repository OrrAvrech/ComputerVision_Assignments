function [rImg] = ReconstructImage(Lpyr)

rImg = sum(Lpyr, 3);

end

