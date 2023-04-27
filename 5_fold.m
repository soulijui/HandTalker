k = 5; % number of folds
datastore = imageDatastore(fullfile('.'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

partStores{k} = [];
for i = 1:k
   temp = partition(datastore, k, i);
   partStores{i} = temp.Files;
end

% this will give us some randomization
% though it is still advisable to randomize the data before hand
idx = crossvalind('Kfold', k, k);

for i = 1:k
    test_idx = (idx == i);
    train_idx = ~test_idx;

    test_Store = imageDatastore(partStores{test_idx}, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    train_Store = imageDatastore(cat(1, partStores{train_idx}), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    train_Store.ReadSize = 100;
    targetSize = [224,224];
    a = transform(train_Store,@preproc, IncludeInfo=true);
    dataPreview = preview(a);
    montage(dataPreview(:,1))
    nn=alexnet;
    layers=nn.Layers;
    %the first valuein layers(23) which is 24 represents the number of letters
    %in the alphabet
    layers(23)=fullyConnectedLayer(24,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20);
    layers(24)=softmaxLayer;
    layers(25)=classificationLayer;

    opts=trainingOptions('sgdm','InitialLearnRate',1e-3, 'MaxEpochs',10,'MiniBatchSize',100,'Shuffle','every-epoch', 'ValidationData',imdsValidation,'ValidationFrequency', 3, 'Verbose',false, 'Plots', 'training-progress');
    net{i}=trainNetwork(train_Store,layers,opts);

    pred{i} = classify(net{i}, test_Store);
    accuracy = mean(pred{i} == test_Store.Labels);
    accuracy
end