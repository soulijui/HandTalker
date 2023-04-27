all_images=imageDatastore('improvedtestimgs','IncludeSubfolders',true,'LabelSource','foldernames');
all_images=shuffle(all_images);
all_images.ReadSize = 100;
imdsValidation=imageDatastore('train4','IncludeSubfolders',true,'LabelSource','foldernames');
a = transform(all_images,@preproc, IncludeInfo=true);
dataPreview = preview(a);
montage(dataPreview(:,1))
nn=alexnet;
layers=nn.Layers;
%the first valuein layers(23) which is 24 represents the number of letters
%in the alphabet
layers(23)=fullyConnectedLayer(24,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20);
layers(24)=softmaxLayer;
layers(25)=classificationLayer;

opts=trainingOptions('sgdm','InitialLearnRate',1e-3, 'MaxEpochs',10,'MiniBatchSize',100,'Shuffle', ...
    'every-epoch','ValidationData',imdsValidation,'ValidationFrequency', 3, 'Verbose',false, 'Plots', ...
    'training-progress', L2Regularization=0.00001);
HandTalkerNetv9=trainNetwork(a,layers,opts);

save ('HandTalkerNetv9.mat', 'HandTalkerNetv9', '-v7.3');