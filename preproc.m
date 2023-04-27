function [dataOut,info] = preproc(dataIn,info)
dataOut = cell([size(dataIn,1),2]);
for idx = 1:size(dataIn,1)
      temp = dataIn{idx};

      
      tform = randomAffine2d(Rotation=[-5 5],XTranslation=[-50 50],YTranslation=[-50 50]); 
      outputView = affineOutputView(size(temp),tform);
      temp = imwarp(temp,tform,OutputView=outputView); 

      

      temp = jitterColorHSV(temp, Brightness=[-0.3 0.3],Contrast=[1.2 1.5],Saturation=[-0.4 0.4]);

      sigma = 1+2*rand; 
      temp = imgaussfilt(temp,sigma); 


      if rand<= 0.6
        temp = repmat(rgb2gray(temp),[1 1 3]);
      end

      temp = imnoise(temp, "gaussian");

    dataOut (idx,:) = {temp,info.Label(idx)};

end

end