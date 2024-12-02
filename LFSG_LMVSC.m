
%% LFSG_LMVSC
% I. Kopriva 2024-10

%%
clear all
close all

% Set path to all subfolders
addpath(genpath('.'));

%% parameters
hyper_flag = 1; % 1 - max ACC criterion; 2 - nax NMI criterion; 3 - max F1 criterion

% Number of evaluations
numit =25;
rel_err_thr = 0.001;
split = 1; % 0 - whole dataset; 1 - take randomly 70% of samples per category

%% Load the data from the "Handwritten_numerals" dataset

%dataset = 'Handwritten_numerals';
dataset = 'BBC';
%dataset = 'Caltech101-20';

if strcmp(dataset,'Handwritten_numerals')
    load Handwritten_numerals.mat;
    Y=labels;
    X=data;
    nc=length(unique(Y));
    % Hyperparameters
    alpha = [1e-4 1e-3 1e-2 1e-1 1 10];
    M = [nc 15 25 50];  % numanchors
elseif strcmp(dataset,'BBC')
    load BBC.mat
    Y = truelabel{1};
    nc=length(unique(Y));
    for v=1:length(data)
        X{v}=data{v}';
    end
    alpha = [1 10 20 30];
    M = [nc 10 20 30 40];  % numanchors
elseif strcmp(dataset,'Caltech101-20')
     load Caltech101-20.mat
     nc = length(unique(Y));
     alpha = [0.001 0.01 0.1 1 10];
     M = [nc 50 100];  % numanchors
end

 % Performance metrics are saved in *.mat file named according to:
% save LFSG_LMVSC_dataset_metric_ACC_or_NMI (depends on the value of hyper_flag)

nv=length(X);

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

    % Tuning over M for a fixed alpha
    alpha_tune = alpha(ceil(length(alpha)/2));

    for i = 1:length(M)
        % Perform K-Means on each view
        parfor j=1:nv
            rand('twister',5489);
            [~, H{j}] = litekmeans(X{j},M(i),'MaxIter', 100,'Replicates',10);
        end

        % Core part of this code (LMVSC)
        [F,A(i,:)] = lmv(X',Y,H,alpha_tune);

        % Performance evaluation of clustering result
        A(i,:) = bestMap(Y,A(i,:));
        ACC_o(i) = 1 - computeCE(A(i,:),Y);  % ORACLE
        NMI_o(i) = compute_nmi(A(i,:),Y);
        F1(i) = compute_f(A(i,:),Y);
    end

    [accmax imax] = max(ACC_o);
    M_accmax(it) = M(imax);

    [nmimax imax] = max(NMI_o);
    M_nmimax(it)= M(imax);

    [f1max imax] = max(F1);
    M_f1max(it)= M(imax);

    ACC = zeros(length(M),length(M));
    NMI = zeros(length(M),length(M));
    Fscore = zeros(length(M),length(M));

    for i=1:length(M)-1
        for j=i+1:length(M)
            ACC(i,j)  = 1 - computeCE(A(i,:),A(j,:));
            NMI(i,j) = compute_nmi(A(i,:),A(j,:));
            Fscore(i,j) = compute_f(A(i,:),A(j,:));
        end
    end

    if hyper_flag == 1
        [amax imax] = max(ACC);
        [amax jmax] = max(amax);
    elseif hyper_flag == 2
        [nmix imax] = max(NMI);
        [nmix jmax] = max(nmix);
    end

    M_i = M(imax(jmax));
    labels_i = A(imax(jmax),:);
    M_j = M(jmax);
    labels_j = A(jmax,:);

    M_k1 = round((2*M_i + M_j)/3);
    M_k2 = round((M_i + 2*M_j)/3);

    rel_err = (M_j-M_i)/M_j;
    while rel_err > rel_err_thr

        % Perform K-Means on each view
        parfor i=1:nv
            rand('twister',5489);
            [~, H{i}] = litekmeans(X{i},M_k1,'MaxIter', 100,'Replicates',10);
        end

        % Core part of this code (LMVSC)
        [F,labels_k1] = lmv(X',Y,H,alpha_tune);

        % Perform K-Means on each view
        parfor i=1:nv
            rand('twister',5489);
            [~, H{i}] = litekmeans(X{i},M_k2,'MaxIter', 100,'Replicates',10);
        end

        % Core part of this code (LMVSC)
        [F,labels_k2] = lmv(X',Y,H,alpha_tune);

        labels_i = bestMap(Y,labels_i);
        labels_j = bestMap(Y,labels_j);
        labels_k1 = bestMap(Y,labels_k1);
        labels_k2 = bestMap(Y,labels_k2);

        if hyper_flag == 1
            metric_ik1 = 1 - computeCE(labels_i,labels_k1);;
            metric_k1k2 = 1 - computeCE(labels_k1,labels_k2);
            metric_k2j = 1 - computeCE(labels_k2,labels_j);
        elseif hyper_flag == 2           
            metric_ik1 = compute_nmi(labels_i,labels_k1);            
            metric_k1k2 = compute_nmi(labels_k1,labels_k2);           
            metric_k2j = compute_nmi(labels_k2,labels_j);
        end

        if (metric_ik1 >= metric_k1k2) && (metric_ik1 >= metric_k2j)
            M_j = M_k1;
            labels_j = labels_k1;
            M_k1 = round((2*M_i + M_j)/3);
            M_k2 = round((M_i + 2*M_j)/3);
            rel_err=(M_k1-M_i)/M_k1;
        elseif metric_k1k2 >= metric_k2j
            M_i = M_k1;
            M_j = M_k2;
            labels_i = labels_k1;
            labels_j = labels_k2;
            M_k1 = round((2*M_i + M_j)/3);
            M_k2 = round((M_i + 2*M_j)/3);
            rel_err=(M_k2-M_k1)/M_k2;
        else
            M_i = M_k2;
            labels_i = labels_k2;
            M_k1 = round((2*M_i + M_j)/3);
            M_k2 = round((M_i + 2*M_j)/3);
            rel_err=(M_j-M_k2)/M_j;
        end
    end

    if hyper_flag == 1
        metric_ik1 = 1 - computeCE(labels_i,labels_k1);
        metric_ik2 = 1 - computeCE(labels_k2,labels_j);
    elseif hyper_flag == 2
        metric_ik1 = compute_nmi(labels_i,labels_k1);
        metric_ik2 = compute_nmi(labels_k2,labels_j);
    end

    if metric_ik1 >= metric_ik2
        M_k = M_k1;
    else
        M_k = M_k2;
    end

    M_est(it)=M_k;

    if hyper_flag == 1
        M_star = M_accmax(it);
    elseif hyper_flag == 2
        M_star = M_nmimax(it);
    end

    clear ACC_o NMI_o F1

    % tuning over alpha for a fixed "M"
    for i=1:length(alpha)
        % Perform K-Means on each view
        parfor j=1:nv
            rand('twister',5489);
            [~, H{j}] = litekmeans(X{j},M_star,'MaxIter', 100,'Replicates',10);
        end

        % Core part of this code (LMVSC)
        [F,A(i,:)] = lmv(X',Y,H,alpha(i));

        A(i,:) = bestMap(Y,A(i,:));
        ACC_o(i)  = 1-computeCE(A(i,:),Y);
        NMI_o(i) = compute_nmi(A(i,:),Y);
        F1(i) = compute_f(A(i,:),Y);
    end

    [accmax imax] = max(ACC_o);
    alpha_accmax(it) = alpha(imax);

    [nmimax imax] = max(NMI_o);
    alpha_nmimax(it)= alpha(imax);

    [f1max imax] = max(F1);
    alpha_f1max(it)= alpha(imax);

    ACC = zeros(length(alpha),length(alpha));
    NMI = zeros(length(alpha),length(alpha));
    Fscore = zeros(length(alpha),length(alpha));

    for i=1:length(alpha)-1
        for j=i+1:length(alpha)
            ACC(i,j)  = 1 - computeCE(A(i,:),A(j,:));
            NMI(i,j) = compute_nmi(A(i,:),A(j,:));
            Fscore(i,j) = compute_f(A(i,:),A(j,:));
        end
    end

    if hyper_flag == 1
        [amax imax] = max(ACC);
        [amax jmax] = max(amax);
    elseif hyper_flag == 2
        [nmix imax] = max(NMI);
        [nmix jmax] = max(nmix);
    end

    alpha_i = alpha(imax(jmax));
    labels_i = A(imax(jmax),:);
    alpha_j = alpha(jmax);
    labels_j = A(jmax,:);

    alpha_k1 = (2*alpha_i + alpha_j)/3;
    alpha_k2 = (alpha_i + 2*alpha_j)/3;

    rel_err = (alpha_j-alpha_i)/alpha_j;

    while rel_err > rel_err_thr
        % Perform K-Means on each view
        parfor i=1:nv
            rand('twister',5489);
            [~, H{i}] = litekmeans(X{i},M_k,'MaxIter', 100,'Replicates',10);
        end

        % Core part of this code (LMVSC)
        [F,labels_k1] = lmv(X',Y,H,alpha_k1);

        % Perform K-Means on each view
        parfor i=1:nv
            rand('twister',5489);
            [~, H{i}] = litekmeans(X{i},M_k,'MaxIter', 100,'Replicates',10);
        end

        % Core part of this code (LMVSC)
        [F,labels_k2] = lmv(X',Y,H,alpha_k2);

        labels_i = bestMap(Y,labels_i);
        labels_j = bestMap(Y,labels_j);
        labels_k1 = bestMap(Y,labels_k1);
        labels_k2 = bestMap(Y,labels_k2);

        if hyper_flag == 1
            metric_ik1 = 1 - computeCE(labels_i,labels_k1);
            metric_k1k2 = 1 - computeCE(labels_k1,labels_k2);
            metric_k2j = 1 - computeCE(labels_k2,labels_j);
        elseif hyper_flag == 2
            metric_ik1 = compute_nmi(labels_i,labels_k1);
            metric_k1k2 = compute_nmi(labels_k1,labels_k2);
            metric_k2j = compute_nmi(labels_k2,labels_j);
        end

        if (metric_ik1 >= metric_k1k2) && (metric_ik1 >= metric_k2j)
            alpha_j = alpha_k1;
            labels_j = labels_k1;
            alpha_k1 = (2*alpha_i + alpha_j)/3;
            alpha_k2 = (alpha_i + 2*alpha_j)/3;
            rel_err=(alpha_k1-alpha_i)/alpha_k1;
        elseif metric_k1k2 >= metric_k2j
            alpha_i = alpha_k1;
            alpha_j = alpha_k2;
            labels_i = labels_k1;
            labels_j = labels_k2;
            alpha_k1 = (2*alpha_i + alpha_j)/3;
            alpha_k2 = (alpha_i + 2*alpha_j)/3;
            rel_err=(alpha_k2-alpha_k1)/alpha_k2;
        else
            alpha_i = alpha_k2;
            labels_i = labels_k2;
            alpha_k1 = (2*alpha_i + alpha_j)/3;
            alpha_k2 = (alpha_i + 2*alpha_j)/3;
            rel_err=(alpha_j-alpha_k2)/alpha_j;
        end
    end

    if hyper_flag == 1
        metric_ik1 = 1 - computeCE(labels_i,labels_k1);
        metric_ik2 = 1 - computeCE(labels_k2,labels_j);
    elseif hyper_flag == 2
        metric_ik1 = compute_nmi(labels_i,labels_k1);
        metric_ik2 = compute_nmi(labels_k2,labels_j);
    end

    if metric_ik1 >= metric_ik2
        alpha_k = alpha_k1;
    else
        alpha_k = alpha_k2;
    end

    alpha_est(it)=alpha_k;

    if hyper_flag == 1
       % Perform K-Means on each view
        parfor i=1:nv
            rand('twister',5489);
            [~, H{i}] = litekmeans(X{i},M_accmax(it),'MaxIter', 100,'Replicates',10);
        end
        % Core part of this code (LMVSC)
        [F,labels_star] = lmv(X',Y,H,alpha_accmax(it));
    elseif hyper_flag == 2
        % Perform K-Means on each view
        parfor i=1:nv
            rand('twister',5489);
            [~, H{i}] = litekmeans(X{i},M_nmimax(it),'MaxIter', 100,'Replicates',10);
        end
        % Core part of this code (LMVSC)
        [F,labels_star] = lmv(X',Y,H,alpha_nmimax(it));
    end

    labels_star = bestMap(Y,labels_star);

    ACC_star(it) = 1 - computeCE(labels_star,Y);
    NMI_star(it) = compute_nmi(labels_star,Y);
    F1_star(it) = compute_f(labels_star',Y);


    % Perform K-Means on each view
    parfor i=1:nv
        rand('twister',5489);
        [~, H{i}] = litekmeans(X{i},M_k,'MaxIter', 100,'Replicates',10);
    end
    % Core part of this code (LMVSC)
    [F,labels_est] = lmv(X',Y,H,alpha_k);

     labels_est = bestMap(Y,labels_est);

    CE_est  = computeCE(labels_est,Y);
    ACC_est(it) = 1 - CE_est;
    NMI_est(it) = compute_nmi(labels_est,Y);
    F1_est(it) = compute_f(labels_est,Y');

    clear ACC_o NMI_o F1

    save LFSG_LMVSC_BBC_metric_ACC.mat ACC_star ACC_est NMI_star NMI_est F1_star F1_est ...
    M_est M_accmax M_nmimax M_f1max ...
    alpha_est alpha_accmax alpha_nmimax alpha_f1max
end

display('********** ACCURACY *****************')
display('ORACLE')
display('Mean')
mean(ACC_star)
display('Std:')
std(ACC_star)

display('ESTIMATED - IN-SAMPLE DATA')
display('Mean')
mean(ACC_est)
display('Std:')
std(ACC_est)

% ranksum two sided Wilcoxon test of statistical significance
p_acc_in = ranksum(ACC_star,ACC_est)

display('*************** NMI ********************')
display('ORACLE')
display('Mean:')
mean(NMI_star)
display('Std:')
std(NMI_star)

display('ESTIMATED: IN-SMAPLE')
display('Mean')
mean(NMI_est)
display('Std:')
std(NMI_est)

% ranksum two sided Wilcoxon test of statistical significance
p_nmi_in = ranksum(NMI_star,NMI_est)

display('*************** F1 score ********************')
display('ORACLE')
display('Mean:')
mean(F1_star)
display('Std:')
std(F1_star)

display('ESTIMATED: IN-SAMPLE')
display('Mean')
mean(F1_est)
display('Std:')
std(F1_est)

% ranksum two sided Wilcoxon test of statistical significance
p_F1_in = ranksum(F1_star,F1_est)

display('***************** " M " **********************')
display('MAXACC ORACLE:')
mean(M_accmax)
std(M_accmax)
display('MAXNMI ORACLE:')
mean(M_nmimax)
std(M_nmimax)
display('MAXF1 ORACLE:')
mean(M_f1max)
std(M_f1max)
display('M_EST:')
mean(M_est)
std(M_est)

% ranksum two sided Wilcoxon test of statistical significance
p_M_acc = ranksum(M_accmax,M_est)
p_M_nmi = ranksum(M_nmimax,M_est)
p_M_F1 = ranksum(M_f1max,M_est)

display('***************** alpha **********************')
display('MAXACC ORACLE:')
mean(alpha_accmax)
std(alpha_accmax)
display('MAXNMI ORACLE:')
mean(alpha_nmimax)
std(alpha_nmimax)
display('MAXF1 ORACLE:')
mean(alpha_f1max)
std(alpha_f1max)
display('alpha_EST:')
mean(alpha_est)
std(alpha_est)

% ranksum two sided Wilcoxon test of statistical significance
p_alpha_acc = ranksum(alpha_accmax,alpha_est)
p_alpha_nmi = ranksum(alpha_nmimax,alpha_est)
p_alpha_F1 = ranksum(alpha_f1max,alpha_est)

save LFSG_LMVSC_BBC_metric_ACC.mat ACC_star ACC_est NMI_star NMI_est F1_star F1_est ...
    M_est M_accmax M_nmimax M_f1max p_M_F1 p_M_nmi p_M_acc...
    alpha_est alpha_accmax alpha_nmimax alpha_f1max p_alpha_F1 p_alpha_nmi p_alpha_acc
