# #######################################################################################################################
# The code below is for the empirical study presented in the document, and follows the method and code at 
https://github.com/GlenMartin31/Multivariate-Binary-CPMs/blob/master/Code/00_Simulation_Functions.R
# #######################################################################################################################

####-----------------------------------------------------------------------------------------
require(tidyverse)
require(pROC)
require(rjags)
require(coda)
require(pbivnorm)
require(nnet)
require(VGAM)
require(glmnet)
require(pbivnorm)
####-----------------------------------------------------------------------------------------
## Step one: Pre-process data
The code below pre-processes the data in the following ways. 
1.) Rename variables and Encode 'Sex' into 0/1.
2.) Randomly split the data into training and validation set
3.) Define IPD and Validation.Population set
####-----------------------------------------------------------------------------------------
Data <- read_csv("/Users/wangyuqi/Downloads/DevelopmentData(1).csv")
head(Data)
data<-rename(Data, "X1" = "Sex", "X2"="Smoking_Status","Y1"="Diabetes","Y2"="CKD")
data$X1<-ifelse(data$X1=="M",1,0)
head(data)
data_1 <-data[2:5]
head(data_1)
CombinedData <-mutate(data_1,Y_Categories = fct_relevel(factor(ifelse(Y1 == 0 & Y2 == 0, 
                                                                      "Y00",
                                                                      ifelse(Y1 == 1 & Y2 == 0,
                                                                             "Y10",
                                                                             ifelse(Y1 == 0 & Y2 == 1,
                                                                                    "Y01",
                                                                                    "Y11")))),
                                                        c("Y00", "Y10", "Y01", "Y11")))
head(CombinedData)
#Randomly split into an IPD and validation set
#make this example reproducible
set.seed(1)
#Use 70% of dataset as training set and remaining 30% as testing set
sample <- sample(c(TRUE, FALSE), nrow(CombinedData), replace=TRUE, prob=c(0.7,0.3))
training  <- CombinedData[sample, ]
testing <- CombinedData[!sample, ]
##Define the IPD and Validation.Population set
IPD <- mutate(training,
         True_P11 =sum(training$Y_Categories == "Y11") / nrow(training) ,
         True_P10 =sum(training$Y_Categories == "Y10") / nrow(training) ,
         True_P01 =sum(training$Y_Categories == "Y01") / nrow(training) ,
         True_P00 =sum(training$Y_Categories == "Y00") / nrow(training), 
         X0 = 1)
head(IPD)

Validation.Population <-mutate(testing,True_P11 =sum(testing$Y_Categories == "Y11") / nrow(testing) ,
                               True_P10 =sum(testing$Y_Categories == "Y10") / nrow(testing) ,
                               True_P01 =sum(testing$Y_Categories == "Y01") / nrow(testing) ,
                               True_P00 =sum(testing$Y_Categories == "Y00") / nrow(testing),
                               X0 = 1)
head(Validation.Population)  

  
  
  
####-----------------
  
## Step two: Multivariate Probit Model Development
  
####-----------------
BayesianProbitMultivariateModel <- jags.model('/Users/wangyuqi/Downloads/JAGSProbitModel.txt', 
                                                 
                                                 data = list(
                                                   
                                                   "n" = nrow(IPD),
                                                   
                                                   "P" = 3,
                                                   
                                                   "Y" = cbind(IPD$Y1, 
                                                               
                                                               IPD$Y2),
                                                   
                                                   "X" = cbind(1,
                                                               
                                                               IPD$X1, 
                                                               
                                                               IPD$X2),
                                                   
                                                   'b_0' = rep(0,3),
                                                   
                                                   'B_0' = diag(1, ncol = 3, nrow = 3)*0.1 #precision
                                                   
                                                 ), 
                                                 
                                                 inits = list("Z" = cbind(IPD$Y1, 
                                                                          
                                                                          IPD$Y2),
                                                              
                                                              "rho" = 0),
                                                 
                                                 n.chains = 1, 
                                                 
                                                 n.adapt = 1000)
  
#sample from the posterior distribution:
samps <- coda.samples( BayesianProbitMultivariateModel, c('Beta','rho'), n.iter = 10000 )
tidy.samps <- samps[[1]][5001:10000,] #set first 5000 samples as burn-in
post.means <- colMeans(tidy.samps) #take the posterior mean
#Predict the risk of each outcome in the validation cohort using posterior mean
X.Beta.Y1 <- as.numeric(cbind(Validation.Population$X0, 
                                
                                Validation.Population$X1,
                                
                                Validation.Population$X2) %*% post.means[paste("Beta[", 1:3, ",1]", sep = "")])
  
X.Beta.Y2 <- as.numeric(cbind(Validation.Population$X0, 
                                
                                Validation.Population$X1,
                                
                                Validation.Population$X2) %*% post.means[paste("Beta[", 1:3, ",2]", sep = "")])
  
#Calculate the joint risks, which this method obtains directly based on the marginal risks and estimate of rho
  
  Validation.Population$MPM_P11 <- pbivnorm(x = cbind(X.Beta.Y1, X.Beta.Y2), 
                                            
                                            rho = post.means["rho"])
  
  Validation.Population$MPM_P10 <- pbivnorm(x = cbind(X.Beta.Y1, -X.Beta.Y2), 
                                            
                                            rho = -post.means["rho"])
  
  Validation.Population$MPM_P01 <- pbivnorm(x = cbind(-X.Beta.Y1, X.Beta.Y2), 
                                            
                                            rho = -post.means["rho"])
  
  Validation.Population$MPM_P00 <- pbivnorm(x = cbind(-X.Beta.Y1, -X.Beta.Y2),
                                            
                                            rho = post.means["rho"])
 head(Validation.Population) 
####-----------------
  
## Step Three：model validation
  
####-----------------
  
## Extract relevant information from the validation set
Predictions <- Validation.Population %>%
   select(Y1, Y2, Y_Categories,MPM_P11, MPM_P10, MPM_P01, MPM_P00) %>%
   mutate_at(vars(ends_with("_P11"),
                  ends_with("_P01"),
                  ends_with("_P10"),
                  ends_with("_P00")), 
             ##turn very small (practically 0 probs) to small number for entry into calibration models (in VGAM)
             ~ifelse(.<=1e-10, 1e-10, .)) 

##To calculate predictive performance of joint outcome risks
  
    #calibration-in-the-large
    
CalInt.model <- coefficients(vgam(Predictions$Y_Categories ~ 1, 
                                  offset = data.matrix(Predictions %>%
                                                         rename_all(~(sub("([A-Z,a-z]+)_", "", 
                                                                          make.names(names(Predictions))))) %>%
                                                         mutate(P10_Z = log(P10 / P00),
                                                                P01_Z = log(P01 / P00),
                                                                P11_Z = log(P11 / P00)) %>%
                                                         select(P10_Z, P01_Z, P11_Z)),
                                  family = multinomial(refLevel = "Y00")))
CalInt_P10 <- as.numeric(CalInt.model[1])
CalInt_P01 <- as.numeric(CalInt.model[2])
CalInt_P11 <- as.numeric(CalInt.model[3])

#calibration slope
  Predictions_1<-Predictions %>%
    mutate(P10_Z = log(MPM_P10 / MPM_P00),
           P01_Z = log(MPM_P01 / MPM_P00),
           P11_Z = log(MPM_P11 / MPM_P00),
           Y_Categories=Y_Categories) %>%
    select(P10_Z, P01_Z, P11_Z,Y_Categories)
  Predictions_1
  k <- length(levels(Predictions_1$Y_Categories)) #number of outcome categories
  CalSlope.model <- coefficients(vgam(Predictions_1$Y_Categories ~ P10_Z + P01_Z + P11_Z, 
                                      data = Predictions_1 %>%
                                        select(P10_Z, P01_Z, P11_Z),
                                      family = multinomial(refLevel = "Y00"),
                                      constraints = list("(Intercept)" = diag(1, 
                                                                              ncol = (k - 1), 
                                                                              nrow = (k - 1)),
                                                         "P10_Z" = rbind(1, 0, 0),
                                                         "P01_Z" = rbind(0, 1, 0),
                                                         "P11_Z" = rbind(0, 0, 1))))
  CalSlope_P10 <- as.numeric(CalSlope.model["P10_Z"])
  CalSlope_P01 <- as.numeric(CalSlope.model["P01_Z"])
  CalSlope_P11 <- as.numeric(CalSlope.model["P11_Z"])
 
  
  
  #Discrimination: also extract a one-versus-rest AUC:
  
  Y11 <- ifelse(Predictions$Y_Categories == "Y11", 1, 0)
  
  Y10 <- ifelse(Predictions$Y_Categories == "Y10", 1, 0)
  
  Y01 <- ifelse(Predictions$Y_Categories == "Y01", 1, 0)
  
  AUC_P11 <- as.numeric(roc(response = Y11, 
                            predictor = as.vector(Predictions %>%
                                                    
                                                    .$MPM_P11),
                            direction = "<",
                            levels = c(0,1))$auc)
  
  
  AUC_P10 <- as.numeric(roc(response = Y10, 
                            predictor = as.vector(Predictions %>%
                                                    
                                                    .$MPM_P10),
                            direction = "<",
                            levels = c(0,1))$auc)
  
  
  
  AUC_P01 <- as.numeric(roc(response = Y01, 
                            predictor = as.vector(Predictions %>%
                                                    
                                                    .$MPM_P01),
                            direction = "<",
                            levels = c(0,1))$auc)
  
  
  
\
  ## Store performance results in a data.frame 
Results <- data.frame(CalInt_P11, CalInt_P10,CalInt_P01,CalSlope_P11,CalSlope_P10,CalSlope_P01,
                        AUC_P11,AUC_P10,AUC_P01)
Results

