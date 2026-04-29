close all; clear all; clc;

%% Inputs
fileloca = "ARAD10\Polars\";
smoothingparam = 0.7;
debug = true; % to print debug plots or not
verbose = true; % to print EXTRA debug plots or not
output_file = "compiledData-ARAD10.txt";

files = dir(fileloca);
files = files(~[files.isdir]); % eliminate non file files from the list

fitalphas = -20:0.5:20;

fitdatacls = zeros(length(fitalphas), length(files));
fitdatacds = zeros(length(fitalphas), length(files));
fitdataRE = zeros(length(files), 1);
fitdataMA = zeros(length(files), 1);

for ff = 1:length(files)
    % Get the data from that airfoil

    foildata = importdata(strcat(fileloca, files(ff).name));
    
    foilalphas = foildata.data(:, 1);
    foilCls = foildata.data(:, 2);
    foilCds = foildata.data(:, 3);
    
    if debug
        %figure('name', "Airfoil data and curve fits")
        subplot(2, 1, 1)
        plot(foilalphas, foilCls, '*', foilalphas, foilCds, '*')
        xlabel("alpha")
        legend("Cl", "Cd")
        subplot(2, 1, 2)
        plot(foilalphas, foilCls./foilCds, '*')
        xlabel("alpha")
        legend("Cl / Cd")
    end

    % extend the foil data, first get the step size in alpha
    alpha_step_size = mode(foilalphas(2:end)-foilalphas(1:end-1));
    preamble = -90:alpha_step_size:foilalphas(1); preamble = preamble(1:end-1);
    postamble = foilalphas(end):alpha_step_size:90; postamble = postamble(2:end);
    foilalphas = [preamble, foilalphas(1:end).', postamble];
    foilCls = [NaN(size(preamble)), foilCls.', NaN(size(postamble))];
    foilCds = [NaN(size(preamble)), foilCds.', NaN(size(postamble))];
    
    % We have a bunch of nan values filling in our data, lets replace those
    % with either: 1-cos(alpha) with matching cD at limit of XFLR5 data
    % or with: sin(alpha)*max(CL) for CL
    fillerCL = sind(foilalphas.*2) * max(foilCls);
    fillerCD = (1 - cosd(foilalphas.*2)) * max(foilCds) / (1 - cosd(postamble(1).*2));

    % Apply our fillers wherever foil data is nan
    foilCls(isnan(foilCls)) = fillerCL(isnan(foilCls));
    foilCds(isnan(foilCds)) = fillerCD(isnan(foilCds));

    % ensure you have curve fitting toolbox available
    foilClsfit = fit(foilalphas.', foilCls.', 'smoothingspline', 'SmoothingParam', smoothingparam);
    foilCls = feval(foilClsfit, fitalphas);
    
    foilCdsfit = fit(foilalphas.', foilCds.', 'smoothingspline', 'SmoothingParam', smoothingparam);
    foilCds = feval(foilCdsfit, fitalphas);
    
    if debug
        subplot(2, 1, 1)
        hold on
        plot(fitalphas, foilCls, fitalphas, foilCds)
        xlabel("alpha")
        legend("Cl", "Cd")
        xline(0, 'k')
        yline(0, 'k')
        xlim([-20, 20])
        ylim([-2, 2])

        subplot(2, 1, 2)
        hold on
        plot(fitalphas, foilCls./foilCds)
        xlabel("alpha")
        legend("Cl / Cd")
        xline(0, 'k')
        yline(0, 'k')
        xlim([-20, 20])
        ylim([-20, 100])
    end
    
    fitdatacls(:, ff) = foilCls;
    fitdatacds(:, ff) = foilCds;
    
    mIndex = strfind(files(ff).name, "_M");
    mIndex = mIndex(end); % Get the last _m in the name (will always be mach number)
    rIndex = strfind(files(ff).name, "_Re");
    rIndex = rIndex(end); % Get the last _Re in the name (will always be mach number)

    fitdataMA(ff) = str2double(files(ff).name(mIndex+2:mIndex+5));
    fitdataRE(ff) = str2double(files(ff).name(rIndex+3:rIndex+7))*1000000;
end

% Need to create 5 column vectors, one for each of CL, CD, alpha,
% reynolds and mach, where each row corresponds to a unique point in
% the provided data above

% Fitalphas get repeated for every set of points
fitalphas = repmat(fitalphas, length(files), 1);

% Re and Ma also get repeated for every point, but, in the other
% direction

% Need to now interpolate in 3D
fitdataRE = repmat(fitdataRE, 1, length(foilCls));
fitdataMA = repmat(fitdataMA, 1, length(foilCls));

% Somehow i have all these matrices transposed, so sneaky transpose to
% correct ordering
fitalphas = fitalphas.';
fitdataRE = fitdataRE.';
fitdataMA = fitdataMA.';

% Reshape all of these matrices into vectors
veclength = numel(fitdatacls);
fitdatacls = reshape(fitdatacls, veclength, 1);
fitdatacds = reshape(fitdatacds, veclength, 1);
fitalphas = reshape(fitalphas, veclength, 1);
fitdataRE = reshape(fitdataRE, veclength, 1);
fitdataMA = reshape(fitdataMA, veclength, 1);

% Use the interpolant to create a dense grid of data

% interpolate our data into the grid

% create our starting grid:
[newalf, newrey, newmac] = meshgrid(unique(fitalphas), unique(fitdataRE), unique(fitdataMA));

% interpolate our data into the grid
gridded_cl = griddata(fitalphas, fitdataRE, fitdataMA, fitdatacls, newalf, newrey, newmac, "nearest");
gridded_cd = griddata(fitalphas, fitdataRE, fitdataMA, fitdatacds, newalf, newrey, newmac, "nearest");

% create a scattered interpolant of the available data (for some slower
% calculation)
cl_scat = scatteredInterpolant(fitalphas, fitdataRE, fitdataMA, fitdatacls, "natural");
cd_scat = scatteredInterpolant(fitalphas, fitdataRE, fitdataMA, fitdatacds, "natural");

% Fill in the nan values where they exist
nanvalues = isnan(gridded_cl) & isnan(gridded_cd);
gridded_cl(nanvalues) = cl_scat(newalf(nanvalues), newrey(nanvalues), newmac(nanvalues));
gridded_cd(nanvalues) = cd_scat(newalf(nanvalues), newrey(nanvalues), newmac(nanvalues));

% contourslice(newalf, newrey, newmac, gridded_cl, [], [], [0.05, 0.25, 0.45], [-1:0.05:2])
% xlabel("alpha"); ylabel("reynolds"); zlabel("mach");
% view(3);
% xlim([-5, 20]);

% Convert from meshgrid to ndgrid format (fml)
P = [2 1 3];
newalf = permute(newalf, P);
newrey = permute(newrey, P);
newmac = permute(newmac, P);
gridded_cl = permute(gridded_cl, P);
gridded_cd = permute(gridded_cd, P);

% Export the gridded CL, CD and input alpha, reynolds and mach numbers
writematrix(newalf, output_file, 'Delimiter', ',', 'WriteMode', 'overwrite');
writematrix(newrey, output_file, 'Delimiter', ',', 'WriteMode', 'append');
writematrix(newmac, output_file, 'Delimiter', ',', 'WriteMode', 'append');
writematrix(gridded_cl, output_file, 'Delimiter', ',', 'WriteMode', 'append');
writematrix(gridded_cd, output_file, 'Delimiter', ',', 'WriteMode', 'append');