DIR = dir('train_set');
X = [];
C = [];
for i = 3:length(DIR)
    dirname = strcat('train_set/',DIR(i).name);
    disp(dirname);
    AUX_DIR = dir(dirname);
    for j = 3:length(AUX_DIR)
        auxfilename = strcat(dirname,'/',AUX_DIR(j).name);
        aux = double(imread(auxfilename))/255.0;
        dim = size(aux);
        X = [X ; reshape(aux,[1,numel(aux)])];
        C = [C ; i-3];
    end
end

mX = mean(X,1);
Y = X - repmat(mX,[size(X,1),1]);
sX = mean(Y.^2,1);
Z = Y ./ repmat(sqrt(sX),[size(X,1),1]);

K = 10;
D = 10;
[W,b] = fuzzy_kflats_clustering(Z,K,D,1.5,25);

figure;
for k = 1:K
    for d = 1:D
        aux = W(k,:,d);
        aux = (aux-min(aux))/(max(aux)-min(aux));
        subplot(K,D,D*(k-1)+d);
        imshow(kron(reshape(aux,dim),ones(8,8)));
    end
end