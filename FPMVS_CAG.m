%
% FPMVS_CAG
%
% I. Kopriva 09-2024

clear all
close all
clc;
warning off;
addpath(genpath('./'));

% Number of evaluations
numit =25;
split = 1; % 0 - whole dataset; 1 - take randomly 70% of samples per category

%% dataset
%dataName = 'Caltech101-20';
% dataName = 'BBC';
 dataName = 'Handwritten_numerals';

if strcmp(dataName,'Caltech101-20')
    dsPath = './FPMVS-CAG-IEEE-TIP_2022/0-dataset/';
    load(strcat(dsPath,dataName));
    k = length(unique(Y));
    % para setting
    anchor = k;
    d = k ;
    lambda=0;
elseif strcmp(dataName,'Handwritten_numerals')
    load Handwritten_numerals.mat;
    Y=labels;
    X=data;
    k=length(unique(Y));
    % para setting
    anchor = k;
    d = k ;
    lambda=0;

elseif strcmp(dataName,'BBC')
    load BBC.mat
    Y = truelabel{1}';
    X = data; clear data
    k=length(unique(Y));
    % para setting
    anchor = k;
    d = k ;
    lambda=0;

    tmp=cell(4,1);
    for i=1:4
        tmp{i}=X{i}';
    end
    X=tmp;
    clear tmp;
end

XX = X;
YY = Y; 

%% Iterate
for it=1:numit
    fprintf('Iter %d\n',it);

    X = XX; Y = YY;
    if split == 1
        % Each category is separately split, to ensure proportional representation
        nIn = 1;
        if strcmp(dataset,'Handwritten_numerals')
            for c=1:nc % Through all categories
                ind = (Y == c); % Indices of the chosen category
                tmp=X{1};
                tmp_c = tmp(ind,:);
                numSamples = size(tmp_c,1);
                numIn = ceil(numSamples*0.7); % 70% of data per category for in-sample set
                indp = randperm(numSamples); % Random permutation of the indices
                tmp1_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                tmp=X{2};
                tmp_c = tmp(ind,:);
                tmp2_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                tmp=X{3};
                tmp_c = tmp(ind,:);
                tmp3_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                tmp=X{4};
                tmp_c = tmp(ind,:);
                tmp4_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                tmp=X{5};
                tmp_c = tmp(ind,:);
                tmp5_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                tmp=X{6};
                tmp_c = tmp(ind,:);
                tmp6_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                labels_in( nIn:nIn + numIn-1) = c; % Labels
                nIn  = nIn  + numIn; % Next indices labels_i
            end
            X{1} = tmp1_in; X{2} = tmp2_in; X{3} = tmp3_in; X{4} = tmp4_in;
            X{5} = tmp5_in; X{6} = tmp6_in;
            Y = labels_in;
        elseif strcmp(dataset,'BBC')
            for c=1:nc % Through all categories
                ind = (Y == c); % Indices of the chosen category
                tmp=X{1};
                tmp_c = tmp(ind,:);
                numSamples = size(tmp_c,1);
                numIn = ceil(numSamples*0.7); % 70% of data per category for in-sample set
                indp = randperm(numSamples); % Random permutation of the indices
                tmp1_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                tmp=X{2};
                tmp_c = tmp(ind,:);
                tmp2_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                tmp=X{3};
                tmp_c = tmp(ind,:);
                tmp3_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                tmp=X{4};
                tmp_c = tmp(ind,:);
                tmp4_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                labels_in( nIn:nIn + numIn-1) = c; % Labels
                nIn  = nIn  + numIn; % Next indices labels_i
            end
            X{1} = tmp1_in; X{2} = tmp2_in; X{3} = tmp3_in; X{4} = tmp4_in;
            Y = labels_in;
        elseif strcmp(dataset,'Caltech101-20')
            for c=1:nc % Through all categories
                ind = (Y == c); % Indices of the chosen category
                tmp=X{1};
                tmp_c = tmp(ind,:);
                numSamples = size(tmp_c,1);
                numIn = ceil(numSamples*0.7); % 70% of data per category for in-sample set
                indp = randperm(numSamples); % Random permutation of the indices
                tmp1_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                tmp=X{2};
                tmp_c = tmp(ind,:);
                tmp2_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                tmp=X{3};
                tmp_c = tmp(ind,:);
                tmp3_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                tmp=X{4};
                tmp_c = tmp(ind,:);
                tmp4_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                tmp=X{5};
                tmp_c = tmp(ind,:);
                tmp5_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                tmp=X{6};
                tmp_c = tmp(ind,:);
                tmp6_in(nIn:nIn+numIn-1,:) = tmp_c(indp(1:numIn),:); % Data
                labels_in( nIn:nIn + numIn-1) = c; % Labels
                nIn  = nIn  + numIn; % Next indices labels_i
            end
            X{1} = tmp1_in; X{2} = tmp2_in; X{3} = tmp3_in; X{4} = tmp4_in;
            X{5} = tmp5_in; X{6} = tmp6_in;
            Y = labels_in;
        end
    end

    % LMVSC algorithm
    [U,A,W,Z,iter,obj,alpha,P] = algo_qp(X,Y,lambda,d,anchor); % X,Y,lambda,d,numanchor
    res = myNMIACCwithmean(U,Y,k); % [ACC nmi Purity Fscore Precision Recall AR Entropy]
    ACC(it)=res(1)
    NMI(it)=res(2)
    F1_score(it)=res(4)
end

display('********** ACCURACY *****************')
display('Mean')
mean(ACC)
display('Std:')
std(ACC)

display('********** NMI *****************')
display('Mean')
mean(NMI)
display('Std:')
std(NMI)

display('********** F1 score *****************')
display('Mean')
mean(F1_score)
display('Std:')
std(F1_score)

save FPMVS-CAG_Handwritten_numerals ACC NMI F1_score 
