function write_results_to_file_v2(metric,num_bands,tables)
% Author: Kinan ABBAS
% Creation date: Nov 9 2022
tables.SIR.Properties.VariableNames=["SNR","Prec SPA","VCA","Post Prec SPA","XRAY","RSPA","SPA","POST SPA"];
tables.MER.Properties.VariableNames=["SNR","Prec SPA","VCA","Post Prec SPA","XRAY","RSPA","SPA","POST SPA"];
tables.PSNR.Properties.VariableNames=["SNR","Prec SPA","VCA","Post Prec SPA","XRAY","RSPA","SPA","POST SPA"];
tables.SAM.Properties.VariableNames=["SNR","Prec SPA","VCA","Post Prec SPA","XRAY","RSPA","SPA","POST SPA"];
timestamp=datetime('now','TimeZone','local','Format','d_MMM_y_HH_MM_SS');
filename='Results/'+string(metric)+'_'+string(num_bands)+'_bands_'+string(timestamp)+'.xls';
writetable(tables.SIR,filename,'Sheet','SIR');
writetable(tables.MER,filename,'Sheet','MER');
writetable(tables.SAM,filename,'Sheet','SAM');
writetable(tables.PSNR,filename,'Sheet','PSNR');
end

