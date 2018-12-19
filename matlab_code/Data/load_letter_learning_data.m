function [ab, be, cd, dc] = load_letter_learning_data(path1, plotTrainingData)
% return data with velocity column because that is how the rest of the code
% works.


    %% load some stuff of interest
    
    
    load ([path1 '/Data/letterA']);
    [atr, atr_mean] = fill_with_velocity(letterAtr);  % training data for A
    [ats, atest_mean] = fill_with_velocity(letterAtst); % test data for A
    clear letterAtr letterAtst
    
    load ([path1 '/Data/letterB']);
    [btr, btr_mean]    = fill_with_velocity(letterBtr);  % training data for B
    [bts, btest_mean]  = fill_with_velocity(letterBtst); % test data for B
    clear letterBtr letterBtst

    load ([path1 '/Data/letterC']);
    [ctr, ctr_mean]  = fill_with_velocity(letterCtr);  % training data for ...
    [cts, ctest_mean]  = fill_with_velocity(letterCtst); % test data for ...
    clear letterCtr letterCtst
    
    load ([path1 '/Data/letterD']);
    [dtr, dtr_mean]    = fill_with_velocity(letterDtr);  % training data for ...
    [dts, dtest_mean]  = fill_with_velocity(letterDtst); % test data for ...
    clear letterDtr letterDtst
    
    load ([path1 '/Data/letterE']);
    [etr, etr_mean]   = fill_with_velocity(letterEtr);  % training data for ...
    [ets, etest_mean] = fill_with_velocity(letterEtst); % test data for ...
    clear letterEtr letterEtst
    
    %% observing A, predicting B
    % training data
    for k=1:length(atr.q)
        abtr.q{k,:} = [atr.q{k}  btr.q{k}];
    end
    % testing data
    for k=1:length(ats.q)
        abts.q{k,:} = [ats.q{k}  bts.q{k}];
    end
    abtr_mean   = [atr_mean    btr_mean];
    abtest_mean = [atest_mean  btest_mean];
    
    
    %% observing B, predicting E
    % training data
    for k=1:length(btr.q)
        betr.q{k,:} = [btr.q{k}  etr.q{k}];
    end
    % testing data
    for k=1:length(bts.q)
        bets.q{k,:} = [bts.q{k}  ets.q{k}];
    end    
    betr_mean   = [btr_mean    etr_mean];
    betest_mean = [btest_mean  etest_mean];
    

    %% observing C, predicting d
    % training data
    for k=1:length(dtr.q)
        cdtr.q{k,:} = [ctr.q{k}  dtr.q{k}];
    end
    % testing data
    for k=1:length(cts.q)
        cdts.q{k,:} = [cts.q{k}  dts.q{k}];
    end        
    cdtr_mean   = [ctr_mean    dtr_mean];
    cdtest_mean = [ctest_mean  dtest_mean];
    
    
    %% observing D, predicting C
    % training data
    for k=1:length(dtr.q)
        dctr.q{k,:} = [dtr.q{k}  ctr.q{k}];
    end
    % testing data
    for k=1:length(dts.q)
        dcts.q{k,:} = [dts.q{k}  cts.q{k}];
    end        
    dctr_mean   = [dtr_mean    ctr_mean];
    dctest_mean = [dtest_mean  ctest_mean];
    
    %% Preparing data output
    ab.tr = abtr;      
    ab.test = abts;
    ab.tr_mean   = abtr_mean;
    ab.test_mean = abtest_mean;
    
    be.tr = betr;      
    be.test = bets;
    be.tr_mean   = betr_mean;
    be.test_mean = betest_mean;
    
    cd.tr = cdtr;      
    cd.test = cdts;
    cd.tr_mean   = cdtr_mean;
    cd.test_mean = cdtest_mean;    
    
    dc.tr = dctr;      
    dc.test = dcts;
    dc.tr_mean   = dctr_mean;
    dc.test_mean = dctest_mean;    
    

    %% Plot stuff of interest
    if ~isempty(plotTrainingData)

        plot_dataVSsample('A', atr, ats, plotTrainingData.test, plotTrainingData.test_savePlot)
        if plotTrainingData.xy
            plot_xy_letter('A', atr, atr_mean, plotTrainingData.xy_savePlot);
        end        

        plot_dataVSsample('B', btr, bts, plotTrainingData.test, plotTrainingData.test_savePlot)
        if plotTrainingData.xy
            plot_xy_letter('B', btr, btr_mean, plotTrainingData.xy_savePlot);
        end                

        plot_dataVSsample('C', ctr, cts, plotTrainingData.test, plotTrainingData.test_savePlot)
        if plotTrainingData.xy
            plot_xy_letter('C', ctr, ctr_mean, plotTrainingData.xy_savePlot);
        end        
        
        plot_dataVSsample('D', dtr, dts, plotTrainingData.test, plotTrainingData.test_savePlot)
        if plotTrainingData.xy
            plot_xy_letter('D', dtr, dtr_mean, plotTrainingData.xy_savePlot);
        end                

        plot_dataVSsample('E', etr, ets, plotTrainingData.test, plotTrainingData.test_savePlot)
        if plotTrainingData.xy
            plot_xy_letter('E', etr, etr_mean, plotTrainingData.xy_savePlot);
        end                
    end
    
    
end


function [newltr, mean_] = fill_with_velocity(ltr)
% adding velocity as NaN values to keep the same format as previous code.
    
        
    for k=1:length(ltr.q)
        putVel =  [ltr.q{k}(:,1)   0.*ltr.q{k}(:,1)./0 ...
                   ltr.q{k}(:,2)   0.*ltr.q{k}(:,1)./0];
        tmp = putVel(2:end,:);
        
        % forcing sampling to 100 points
        xorig     = linspace(1, 100, size(tmp,1));
        xresample = linspace(1, 100);
        tmp2      = interp1(xorig, tmp, xresample );
        
        newltr.q{k,:} = tmp2;
    end

    
    me = zeros(size(newltr.q{1}));
    for k=1:length(newltr.q)
       me = me+ newltr.q{k};
    end
    mean_ = me./length(ltr.q);
    
    if 0
        figurew;
        for k=1:length(ltr.q)
            plot(newltr.q{k}(:,1), newltr.q{k}(:,3), SGRAY(0.5));
            plot(mean_(:,1), mean_(:,3), SRED(2));
        end
    end    
    
end

function [] = plot_dataVSsample(letterName, data, dataTest, plotTestData, saveplot)

    h = figurew([letterName ' training']);
    plotTrajectoryStatistics(data, [1:1:length(data.q{1})], 'b' , [1]);  
    plotTrajectoryStatistics(data, [1:1:length(data.q{1})], 'r' , [3]);
    title([letterName '. blue:X, red:Y gray: test']);
    if plotTestData
        for k=1:length(dataTest.q)
            plot(dataTest.q{k}, SGRAY(0.5) );
        end  
    end
    ylim([-0.2    1.2]);
    
    if saveplot
        plot2svg([letterName '_joint.svg'], h, 'png');
    end
    
end

function [] = plot_xy_letter(letterName, data, data_mean, saveplot)

    h = figurew([letterName ' letter']);
    for k=1:length(data.q)
        plot(data.q{k}(:,1), data.q{k}(:,3), SGRAY(2, 0.8));
    end  
    plot(data_mean(:,1), data_mean(:,3), SBLUE(2));
    axis([-0.2   1   -0.2  1]);
    
    if saveplot
        plot2svg([letterName '_cart.svg'], h, 'png');
    end
    
end



