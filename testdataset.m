datastore = imageDatastore('trains4.5','IncludeSubfolders',true,'LabelSource','foldernames');
for i=1:length(datastore.Files)
    input_images{i,1}= imread(datastore.Files{i,1});
    net_input{i,1}=imresize(input_images{i,1},[227,227]);
    filename=datastore.Files{i,1};
    imwrite(net_input{i,1},filename);
end
