library(missForest)
library(impute)
library(hydroGOF)
library(doParallel)
library(ggplot2)

df <- read.table(file="https://raw.githubusercontent.com/dewaranch/Covid/master/data/processed/abridged_stationary_features.tsv",
                 sep="\t",skip=1)

colnames(df)=c('FIPS', 'State', 'lat', 'lon', 'POP_LATITUDE','POP_LONGITUDE', 'CensusRegionName', 'CensusDivisionName',
               'Rural-UrbanContinuumCode2013', 'PopulationEstimate2018','PopTotalMale2017', 'PopTotalFemale2017', 
               'FracMale2017','PopulationEstimate65+2017', 'PopulationDensityperSqMile2010','CensusPopulation2010',
               'MedianAge2010', '#EligibleforMedicare2018','MedicareEnrollment,AgedTot2017', '3-YrDiabetes2015-17',
               'DiabetesPercentage', 'HeartDiseaseMortality', 'StrokeMortality','Smokers_Percentage', 'RespMortalityRate2014',
               '#FTEHospitalTotal2017',"TotalM.D.'s,TotNon-FedandFed2017", '#HospParticipatinginNetwork2017',
               '#Hospitals', '#ICU_beds', 'dem_to_rep_ratio', 'PopMale<52010','PopFmle<52010', 'PopMale5-92010', 
               'PopFmle5-92010', 'PopMale10-142010','PopFmle10-142010', 'PopMale15-192010', 'PopFmle15-192010',
               'PopMale20-242010', 'PopFmle20-242010', 'PopMale25-292010','PopFmle25-292010', 'PopMale30-342010', 
               'PopFmle30-342010','PopMale35-442010', 'PopFmle35-442010', 'PopMale45-542010','PopFmle45-542010',
               'PopMale55-592010', 'PopFmle55-592010','PopMale60-642010', 'PopFmle60-642010', 'PopMale65-742010',
               'PopFmle65-742010', 'PopMale75-842010', 'PopFmle75-842010','PopMale>842010', 'PopFmle>842010', 
               '3-YrMortalityAge<1Year2015-17','3-YrMortalityAge1-4Years2015-17', '3-YrMortalityAge5-14Years2015-17',
               '3-YrMortalityAge15-24Years2015-17','3-YrMortalityAge25-34Years2015-17','3-YrMortalityAge35-44Years2015-17',
               '3-YrMortalityAge45-54Years2015-17','3-YrMortalityAge55-64Years2015-17','3-YrMortalityAge65-74Years2015-17',
               '3-YrMortalityAge75-84Years2015-17', '3-YrMortalityAge85+Years2015-17','mortality2015-17Estimated',
               'SVIPercentile', 'HPSAShortage','HPSAServedPop', 'HPSAUnderservedPop')

drops <- c('lat','lon','CensusPopulation2010','3-YrDiabetes2015-17','3-YrMortalityAge<1Year2015-17',
           '3-YrMortalityAge1-4Years2015-17','3-YrMortalityAge5-14Years2015-17','3-YrMortalityAge15-24Years2015-17',
           '3-YrMortalityAge25-34Years2015-17','3-YrMortalityAge35-44Years2015-17','3-YrMortalityAge45-54Years2015-17',
           '3-YrMortalityAge55-64Years2015-17','3-YrMortalityAge65-74Years2015-17','3-YrMortalityAge75-84Years2015-17',
           '3-YrMortalityAge85+Years2015-17','mortality2015-17Estimated','HPSAShortage','HPSAServedPop','HPSAUnderservedPop')

df <- df[ , !(names(df) %in% drops)]
df <- df[df$FIPS!=46113,]

registerDoParallel(cores=4)
temp_kn <- data.frame()
temp_rd <- data.frame()
df_compl <- df[complete.cases(df),]
for (i in 1:20) {
  start_time <- Sys.time()
  for (k in 1:5) {
    ki <- 20*k
    ni <- 20*k
    df_sample <- sample_n(df_compl,2000,replace=FALSE)
    df_mis <- prodNA(df_sample,noNA=0.01)
    mat_imp_rf <- missForest(data.matrix(df_mis),ntree=ni,parallelize="forests")
    test_rd <- data.matrix(mat_imp_rf$ximp)
    test_df <- impute.knn(data.matrix(df_mis),k=ki)$data
    test_true <- data.matrix(df_sample)
    temp_kn <- rbind(temp_kn,data.frame(k=ki,NRMSE=mixError(ximp=test_df,xmis=data.matrix(df_mis),xtrue=test_true)))
    temp_rd <- rbind(temp_rd,data.frame(n=ni,NRMSE=mixError(ximp=test_rd,xmis=data.matrix(df_mis),xtrue=test_true)))
    print(Sys.time()-start_time)
  }
}

temp2 <- data.frame()
for (i in 1:20) {
  for (k in 1:20) {
    df_compl_mis <- prodNA(df_compl,noNA=0.01)
    test_df <- impute.knn(test,k=k*10)$data
    temp2 <- rbind(temp2,data.frame(k=k,NRMSE=mixError(ximp=test_df,xmis=df_compl_mis,xtrue=test_true)))
  }
}
rownames(temp2) <- NULL
ggplot(data=temp2)+geom_boxplot(aes(x=as.factor(k),y=NRMSE))
temp2$k <- temp2$k*10
temp3 <- rbind(temp,temp2)
mytheme <- function(title_size=20,axis_size=14,element_size=15) {
  theme_bw() %+replace%
    theme(plot.title=element_text(size=title_size),
          axis.title=element_text(size=axis_size),
          axis.text.x=element_text(size=axis_size),
          axis.text.y=element_text(size=axis_size),
          strip.text.y=element_text(size=element_size,color="black",face="bold",angle=0))}
ggplot(data=temp2)+geom_boxplot(aes(x=as.factor(k/10),y=NRMSE))+xlab("K (divided by 10)")+mytheme()

jasmine <- read.csv("C:/Users/sh777/Documents/OneDrive/Harvard/SP20/BST 263/RandomForestCV.csv")
jasmine$Tree <- jasmine$Tree*10
jasmine <- jasmine[c("Tree","PFC")]
write.table(jasmine,file="CV_PFC.tsv",quote=F,sep="\t",col.names=NA)

ggplot(data=jasmine)+geom_boxplot(aes(x=as.factor(Tree*10),y=PFC))+mytheme()+
  xlab("Number of Trees")+ylab("PFC")+ggtitle("PFC for Random Forest")
jasmine$Tree <- jasmine$Tree*10
CV_d <- jasmine
rownames(temp_kn) <- NULL
rownames(temp_rd) <- NULL

ggplot(data=temp_kn)+geom_boxplot(aes(x=as.factor(k),y=NRMSE))+
  mytheme()+xlab("Number of Neighbors")+ylab("NRMSE")+ggtitle("NRMSE for k-NN")

ggplot(data=temp_rd)+geom_boxplot(aes(x=as.factor(n),y=NRMSE))+mytheme()+
  xlab("Number of Trees")+ylab("NRMSE")+ggtitle("NRMSE for Random Forest")+ylim(0.2,0.275)

write.table(temp_kn,file="CV_KNN.tsv",quote=F,sep="\t",col.names=NA)
write.table(temp_rd,file="CV_RF.tsv",quote=F,sep="\t",col.names=NA)

