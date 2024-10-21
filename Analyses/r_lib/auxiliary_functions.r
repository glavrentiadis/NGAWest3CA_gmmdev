#################################################
# This script contains various auxiliary 
# functions for R
#
#################################################

#libraries
library(sp)
#library(rgdal)

#Anonymous functions
and.array      <- function (...)  Reduce("&", list(...))
min.array      <- function (...)  apply(data.frame(list(...)), 1, min)
max.array      <- function (...)  apply(data.frame(list(...)), 1, max)
mean.plus.sig  <- function (x)  mean(x) + sd(x)
mean.minus.sig <- function (x)  mean(x) - sd(x)
mean.2th.98th  <- function(x) mean_se(x, mult=2)
quant.16th     <- function(x) as.double(quantile(x, .16))
quant.84th     <- function(x) as.double(quantile(x, .84))

#Latlon to utm
LongLatToUTM<-function(lat,lon,zone){
  #' Convert Lat Lon to UTM coordinates
  #' 
  #' Input:
  #'  lat: array with latitude degrees
  #'  lon: array longitude degrees
  #'  zone: UTM zone
  #'  
  #' Output:
  #'  xy_utm: data.frame with id, Xutm, Yutm
  
  xy <- data.frame(ID = 1:length(lon), X = lon, Y = lat)
  coordinates(xy) <- c("X", "Y")
  proj4string(xy) <- CRS("+proj=longlat +datum=WGS84")  ## for example
  xy_utm <- spTransform(xy, CRS(paste("+proj=utm +zone=",zone," +datum=WGS84",sep='')))
  return(as.data.frame(xy_utm))
}

#Unique elements
UniqueIdxInv <- function(data_array){
  #' Unique elements, indices and inverse of data_array
  #' 
  #' Input:
  #'  data_array: input array
  #'  
  #' Output:
  #'  unq: unique data
  #'  idx: indices of unique data
  #'  inv: inverse indices for creating original array

  #number of data
  n_data <-length(data_array)

  #create data data-frame
  df_data <- data.frame(data=data_array)
  #get data-frame with unique data
  df_data_unq <- unique(df_data)
  data_unq    <- df_data_unq$data

  #get indices of unique data values
  data_unq_idx <- strtoi(row.names(df_data_unq))
  
  #get inverse indices
  data_unq_inv  <- array(0,n_data)
  for (k in 1:length(data_unq)){
    #return k for element equal to data_unq[k] else 0
    data_unq_inv <- data_unq_inv + ifelse(data_array %in% data_unq[k],k,0)
  }
  
  #return output
  return(list(unq=data_unq, idx=data_unq_idx, inv=data_unq_inv))
}

#piecewise scaling
piecewise_scl <- function(x, xlim1, xlim2, ylim1, ylim2){
    #evalute piecewise linear function 
    y <- approx(x=c(xlim1,xlim2), y=c(ylim1, ylim2), yleft=ylim1, yright=ylim2, xout=x, method="linear")$y        
    return(y)    
}

