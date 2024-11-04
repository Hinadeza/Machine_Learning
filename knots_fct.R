##################################
############ GAM model
##################################

# Picks number of spatial and temporal knots from mgcv models

pick_knots_mgcv <- function(dat) {

# Degree of freedom = number of unique location x number of years
 df <- dat %>%
    group_by(LONGITUDE, LATITUDE) %>%
    dplyr::slice(1)

  max_kspat <- nrow(df)
  max_ktemp <- length(unique(dat$fYEAR))

  kspat <- seq(30, max_kspat, by = 20) # minium of 30 knots on the spatial dimension
  ktemp <- seq(8, max_ktemp, by = 2)  # minimum of 10 knots of the temporal dimension

  knbre<- expand.grid(kspat,ktemp)

  mod_list <- list()

  for ( i in 1 : nrow(knbre)){

    mod0 <- mgcv::gam(COUNT/TOTAL ~ te(LONGITUDE,LATITUDE, fYEAR, # inputs over which to smooth
                                       bs = c("tp", "cr"), # types of bases
                                       k=c(knbre[i,1],knbre[i,2]), # knot count in each dimension
                                       d=c(2,1)), # (s,t) basis dimension
                      data = dat,
                      control =  gam.control(scalePenalty = FALSE),
                      method = "GCV.Cp", family = binomial("logit"),
                      weights = TOTAL)

    mod_list[[i]] <- cbind(as.numeric(summary(mod0)$r.sq), as.numeric(summary(mod0)$s.table[[1]]), as.numeric(summary(mod0)$sp.criterion),
                           as.numeric(AIC(mod0)), knbre[i,1],knbre[i,2])
  }

  table_result <- do.call(rbind, mod_list)
  colnames(table_result) <- c("Rsquared", "edf", "GCV", "AIC", "kspat", "ktemp")

  ## Criteria
  # edf cannot be greater than degree of freedom
  ## lowest GCV
  ## highest r2
  ## lowest AIC

  table_result <- table_result %>%
    data.frame() %>%
    arrange(desc(Rsquared), GCV, desc(AIC))

  return(table_result)
}
