%% LFSG_MCMLE
% I. Kopriva 2024-09

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
split = 1; % 0 - entire dataset; 1 - take randomly 70% of samples per category

%% Load the data from the BBC dataset
%dataset = 'Handwritten_numerals';
dataset = 'BBC';

if strcmp(dataset,'BBC')
    load BBC.mat;
    X = data;
    Y = truelabel{1};
    nc = length(unique(Y));
    for i = 1:length(X)
        X{i} = X{i}';
        X{i} = tfidf(X{i});
    end
    % Hyperparameters
    alpha = [1e-4 1e-2 1e-1];
    beta = [1e-1 1 1e1 1e2 1e3];
elseif strcmp(dataset,'Handwritten_numerals')
    load Handwritten_numerals.mat;
    Y=labels';
    X=data;
    nc = length(unique(Y));
    alpha = [1e-4 1e-2 1e-1 1];
    beta = [1e-1 1 1e1 1e2 1e3];   
end

% Performance metrics are saved in *.mat file named according to:
% save LFSG_MCLME_dataset_metric_ACC_or_NMI (depends on the value of hyper_flag)

XX = X;
YY = Y; 

for i = 1:length(X)
    % ---------- initilization for Z and F -------- %
    options = [];
    options.NeighborMode = 'KNN';
    options.k = nc;
    options.WeightMode = 'HeatKernel';      % Binary  HeatKernel
    W = constructW(X{i},options);
    W = full(W);
    Z1 = W-diag(diag(W));
    W = (Z1+Z1')/2;
    D{i}= diag(sum(W));
    L0 = D{i} - W;
    L{i} = D{i}^(-0.5)*L0*D{i}^(-0.5);
end

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
        end

        clear W Z1 D L0 L

        for i = 1:length(X)
            % ---------- initilization for Z and F -------- %
            options = [];
            options.NeighborMode = 'KNN';
            options.k = nc;
            options.WeightMode = 'HeatKernel';      % Binary  HeatKernel
            W = constructW(X{i},options);
            W = full(W);
            Z1 = W-diag(diag(W));
            W = (Z1+Z1')/2;
            D{i}= diag(sum(W));
            L0 = D{i} - W;
            L{i} = D{i}^(-0.5)*L0*D{i}^(-0.5);
        end
    end

    % Tuning over beta for a fixed alpha
    clear CE NMI_o F1
    alpha_tune = alpha(ceil(length(alpha)/2));
    for i = 1:length(beta)
        [~,G,s,t] = MCMLE(L,D,nc,beta(i),length(L),alpha_tune,size(X{1},1));
        [~, A(i,:)] = max(G, [], 2);
        CE(i)  = computeCE(A(i,:),Y);
        NMI_o(i) = compute_nmi(A(i,:),Y);
        F1(i) = compute_f(A(i,:),Y);
    end

    [cemin imin] = min(CE);
    beta_accmax(it) = beta(imin);

    [nmimax imax] = max(NMI_o);
    beta_nmimax(it)= beta(imax);

    [f1max imax] = max(F1);
    beta_f1max(it)= beta(imax);

    ACC = zeros(length(beta),length(beta));
    NMI = zeros(length(beta),length(beta));
    Fscore = zeros(length(beta),length(beta));

    for i=1:length(beta)-1
        for j=i+1:length(beta)
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

    beta_i = beta(imax(jmax));
    labels_i = A(imax(jmax),:);
    beta_j = beta(jmax);
    labels_j = A(jmax,:);

    beta_k1 = (2*beta_i + beta_j)/3;
    beta_k2 = (beta_i + 2*beta_j)/3; 

    rel_err = (beta_j-beta_i)/beta_j;

    while rel_err > rel_err_thr
        [~,G,s,t] = MCMLE(L,D,nc,beta_k1,length(L),alpha_tune,size(X{1},1));
        [~, labels_k1] = max(G, [], 2);

        [~,G,s,t] = MCMLE(L,D,nc,beta_k2,length(L),alpha_tune,size(X{1},1));
        [~, labels_k2] = max(G, [], 2);

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
            beta_j = beta_k1;
            labels_j = labels_k1;
            beta_k1 = (2*beta_i + beta_j)/3;
            beta_k2 = (beta_i + 2*beta_j)/3;
            rel_err=(beta_k1-beta_i)/beta_k1;
        elseif metric_k1k2 >= metric_k2j
            beta_i = beta_k1;
            beta_j = beta_k2;
            labels_i = labels_k1;
            labels_j = labels_k2;
            beta_k1 = (2*beta_i + beta_j)/3;
            beta_k2 = (beta_i + 2*beta_j)/3;
            rel_err=(beta_k2-beta_k1)/beta_k2;
        else
            beta_i = beta_k2;
            labels_i = labels_k2;
            beta_k1 = (2*beta_i + beta_j)/3;
            beta_k2 = (beta_i + 2*beta_j)/3;
            rel_err=(beta_j-beta_k2)/beta_j;
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
        beta_k = beta_k1;
    else
        beta_k = beta_k2;
    end

    beta_est(it)=beta_k;

    if hyper_flag == 1
        beta_star = beta_accmax(it);
    elseif hyper_flag == 2
        beta_star = beta_nmimax(it);
    end

    clear CE NMI_o F1
    % tuning over alpha for a fixed "beta"
    for i=1:length(alpha)
        [~,G,s,t] = MCMLE(L,D,nc,beta_star,length(L),alpha(i),size(X{1},1));
        [~, A(i,:)] = max(G, [], 2);
        CE(i)  = computeCE(A(i,:),Y);
        NMI_o(i) = compute_nmi(Y,A(i,:));
        F1(i) = compute_f(A(i,:),Y);
    end

    [cemin imin] = min(CE);
    alpha_accmax(it) = alpha(imin);

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
        [~,G,s,t] = MCMLE(L,D,nc,beta_k,length(L),alpha_k1,size(X{1},1));
        [~, labels_k1] = max(G, [], 2);

        [~,G,s,t] = MCMLE(L,D,nc,beta_k,length(L),alpha_k2,size(X{1},1));
        [~, labels_k2] = max(G, [], 2);

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
    elseif hyperflag == 2
        metric_ik1 = compute_nmi(labels_i,labels_k1)
        metric_ik2 = compute_nmi(labels_k2,labels_j);
    end

    if metric_ik1 >= metric_ik2
        alpha_k = alpha_k1;
    else
        alpha_k = alpha_k2;
    end

    alpha_est(it)=alpha_k;

    if hyper_flag == 1
        [~,G,s,t] = MCMLE(L,D,nc,beta_accmax(it),length(L),alpha_accmax(it),size(X{1},1));
    elseif hyper_flag == 2
        [~,G,s,t] = MCMLE(L,D,nc,beta_nmimax(it),length(L),alpha_nmimax(it),size(X{1},1));
    end
    [~, labels_star] = max(G, [], 2);

    ACC_star(it) = 1 - computeCE(labels_star,Y);
    NMI_star(it) = compute_nmi(labels_star,Y);
    F1_star(it) = compute_f(labels_star,Y');

    [~,G,s,t] = MCMLE(L,D,nc,beta_k,length(L),alpha_k,size(X{1},1));
    [~, labels_est] = max(G, [], 2);

    CE_est  = computeCE(labels_est,Y);
    ACC_est(it) = 1 - CE_est;
    NMI_est(it) = compute_nmi(labels_est,Y);
    F1_est(it) = compute_f(labels_est,Y');

    save LFSG_MCMLE_BBC_metric_ACC ACC_star ACC_est NMI_star NMI_est F1_star F1_est ...
    beta_est beta_accmax beta_nmimax beta_f1max...
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

display('***************** " beta " **********************')
display('MAXACC ORACLE:')
mean(beta_accmax)
std(beta_accmax)
display('MAXNMI ORACLE:')
mean(beta_nmimax)
std(beta_nmimax)
display('MAXF1 ORACLE:')
mean(beta_f1max)
std(beta_f1max)
display('beta_EST:')
mean(beta_est)
std(beta_est)

% ranksum two sided Wilcoxon test of statistical significance
p_beta_acc = ranksum(beta_accmax,beta_est)
p_beta_nmi = ranksum(beta_nmimax,beta_est)
p_beta_F1 = ranksum(beta_f1max,beta_est)

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

save LFSG_MCMLE_BBC_metric_ACC ACC_star ACC_est NMI_star NMI_est F1_star F1_est ...
    beta_est beta_accmax beta_nmimax beta_f1max p_beta_F1 p_beta_nmi p_beta_acc...
    alpha_est alpha_accmax alpha_nmimax alpha_f1max p_alpha_F1 p_alpha_nmi p_alpha_acc
