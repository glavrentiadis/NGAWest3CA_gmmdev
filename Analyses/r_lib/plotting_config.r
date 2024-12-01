base_breaks <- function(n = 10){
  function(x) {
    axisTicks(log10(range(x, na.rm = TRUE)), log = TRUE, n = n)
  }
}
#log breaks
breaks_log       <- 10^(-10:10)
breaks_log_minor <- rep(1:9, 21)*(10^rep(-10:10, each=9))
#plotting theme
theme_set(theme_linedraw())

#text size
size_title <- 15
size_text  <- 12

#color palette
color_dW   <- "#666666"
color_dB   <- "#00BFC4"
color_dBP  <- "#4DAF4A"
color_dS2S <- "#F781BF"
color_dSP  <- "#FFFF33"
color_dT   <- "#F8766D"
