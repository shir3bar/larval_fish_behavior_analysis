##### Install/load requirements #####
install.packages(c('precrec','fitdistrplus','visreg'))
library(precrec)
library(fitdistrplus)
library(visreg)
library(ggplot2)
library('RColorBrewer')
library(ggforce)
library(patchwork)

##### Load data file #####
# Data should have a column with the output of the classification score (neuron output, soft scores) for the positive class (after softmax)
path_to_data = '/Volumes/GoogleDrive/My Drive/PhD/Manuscripts/Feeding behavior automatic identification/AI detection of feeding strikes/Data and analysis/all_data.csv'
data = read.csv(path_to_data)

# visualize score distributions for the negative and positive classes and check which distribution fits them:
neg = data[data$labels==0,]
pos = data[data$labels==1,]
hist(neg$scores,breaks = 50, xlab='Scores', ylab='number of samples',
     xlim=c(0,1),main='Negative class score distribution', cex.lab=1.75, cex.axis=1.75, cex.main=1.75, cex.sub=1.75)

hist(pos$scores,breaks = 50, xlab='Scores', ylab='number of samples', 
     xlim=c(0,1),main='Positive class score distribution', cex.lab=1.75, cex.axis=1.75, cex.main=1.75, cex.sub=1.75)
# Calculate ROC/PRC and calculate AUC:
data_curves <- evalmod(scores = data$scores, labels = data$labels)
plot(data_curves,color='red')
auc(data_curves)

##### Fitting distributions to the data #####
# In this part you will have to get creative, as you will have to look for a distribution that fits your data
# For our data (the sample data brought here as an example) we fit the Weibull distribution to the positive class
# and the Beta distribution to the negative class.
# We use the fitdistr package to check how good our distribution fits and what are its parameters.
get_summary = function(dist,scores)
{
  fit = fitdist(scores,distr =dist)
  dist_summary = summary(pos_fit)
  
}
dist_pos =  'weibull' # change this to the distribution of your choice, check the fitdist documentation for supported distributions
fit_pos = fitdist((1-pos$scores),distr =dist_pos) #we substract the scores from  1 to get left skewed scores, we'll do it again later to return to a right skew
s_pos = summary(fit_pos) # save the summary, we'll need it to get the distribution's parameters
plot(fit_pos) # visually check your fit looks good: the red line should fit the histogram in the top left pane
dist_neg = 'beta' # same as before, change it to the distribution that fits your 
fit_neg = fitdist(neg$scores,distr = dist_neg) 
s_neg = summary(fit_neg) # save the summary, we'll need it to get the distribution's parameters
plot(fit_neg) # visually check your fit looks good: the red line should fit the histogram in the top left pane

##### Simulating a new dataset #####
n_pos = 4500
n_neg = 4500
dist_func_pos = rweibull # change this to the random generator of your selected positive distribution
dist_func_neg = rbeta # change this to the random generator of your selected r+negative distribution
get_stat_mat = function(n_pos, n_neg, s_pos, s_neg, dist_func_pos, dist_func_neg){
  pop_pos=1-(dist_func_pos(n_pos, shape=s_pos$estimate[1], scale = s_pos$estimate[2]))
  pop_neg=dist_func_neg(n_neg, shape1=s_neg$estimate[1], shape2 = s_neg$estimate[2])
  stat.mat=matrix(0,n_pos+n_neg,2)
  stat.mat[1:n_pos,1]=1
  stat.mat[,2]=c(pop_pos,pop_neg)
  neg_samps=which(stat.mat[,2]<0) 
  if (length(neg_samps)>0){stat.mat=stat.mat[-neg_samps,]}
  return(stat.mat)
}


# The random populations (pop_pos, pop_neg) aren't guaranteed to be positive, we want to remove negative samples,
# note this might cause the final dataset to be slightly smaller than n_pos+n_neg
obs.stats = get_stat_mat(n_pos, n_neg, s_pos, s_neg, dist_func_pos, dist_func_neg)

# Calculate the simulated dataset's PRC and ROC curves:
sim_curves <- evalmod(scores = obs.stats[,2], labels = obs.stats[,1])
plot(sim_curves)
auc(sim_curves)

##### Now let's try a few dataset imbalances together: #####
# for each data imbalance specify the number of positive (n_pos) and number of negative (n_neg) samples
# make sure the length of n_neg is the same as n_pos:
n_pos = c(63,63,450,4500) # this example is for 0.14%, 1.4%, 9%, 50% positive class percent
n_neg = c(45000,4500,4500,4500) 
dataset_names = as.character((n_pos/n_neg)*100) #labels for the different imbalances as percent positive class
scores = list()
labels = list()
# Iterate over the imabalances and calculate the new dataset stats, i.e. soft classifier scores and labels
# at each imbalance:
for (i in 1:length(n_pos))
{
  tmp.stats = get_stat_mat(n_pos[i], n_neg[i], s_pos, s_neg, dist_func_pos, dist_func_neg)
  scores[[i]] = tmp.stats[,2]
  labels[[i]] = tmp.stats[,1]
}
# the mmdata function prepares data to be plotted using precrec's functions:
combined_data = mmdata(scores,labels,modnames = dataset_names,dsids=c(1,2,3,4))
combined_curves = evalmod(combined_data) #calculate the curves
plot(combined_curves) #plot curves
auc(combined_curves) # show aucs

##### Custom visualizations ####
# I wanted a better looking plot so I tinkered with the precrec graphics functions 
# to change the text size and colors and add labels of AuROC and AuPRC.
# First some functions, these are slightly modified (or not at all) from the precrec package:
get_pn_info <- function(object) {
  nps <- attr(object, "data_info")[["np"]]
  nns <- attr(object, "data_info")[["nn"]]
  
  is_consistant <- TRUE
  prev_np <- NA
  prev_nn <- NA
  np_tot <- 0
  nn_tot <- 0
  n <- 0
  for (i in seq_along(nps)) {
    np <- nps[i]
    nn <- nns[i]
    
    if ((!is.na(prev_np) && np != prev_np)
        ||  (!is.na(prev_nn) && nn != prev_nn)) {
      is_consistant <- FALSE
    }
    
    np_tot <- np_tot + np
    nn_tot <- nn_tot + nn
    prev_np <- np
    prev_nn <- nn
    n <- n + 1
  }
  
  avg_np <- np_tot / n
  avg_nn <- nn_tot / n
  
  prc_base <- avg_np / (avg_np + avg_nn)
  
  list(avg_np = avg_np, avg_nn = avg_nn, is_consistant = is_consistant,
       prc_base = prc_base)
  
}


geom_basic <- function(p, main, xlab, ylab, show_legend) {
  p <- p + ggplot2::theme_classic()
  p <- p + ggplot2::ggtitle(main)
  p <- p + ggplot2::xlab(xlab)
  p <- p + ggplot2::ylab(ylab)
  p <- p + ggplot2::theme(
    legend.title = element_text(size=12), # modify this to change the font size in the legend
    text = element_text(size=16), # modify this to change the font size in general
    plot.title = element_text(size=14,face='bold', hjust = 0.5), )# modify this to change the font size in the title
  if (!show_legend) {
    p <- p + ggplot2::theme(legend.position = "none")
  }
  
  p
}

make_rocprc_title <- function(object, pt) {
  pn_info <- get_pn_info(object)
  np <- pn_info$avg_np
  nn <- pn_info$avg_nn
  paste0(pt, " - P: ", np, ", N: ", nn)
}

geom_basic_roc <- function(p, object, show_legend = TRUE, add_np_nn = TRUE,
                           xlim, ylim, ratio, ...) {
  
  pn_info <- get_pn_info(object)
  
  if (add_np_nn && pn_info$is_consistant) {
    main <- make_rocprc_title(object, "ROC")
  } else {
    main <- "ROC"
  }
  
  p <- p + ggplot2::geom_abline(intercept = 0, slope = 1,colour = "red",
                                linetype = 'dashed', line)
  p <- set_coords(p, xlim, ylim, ratio)
  p <- geom_basic(p, main, "1 - Specificity", "Sensitivity", show_legend)
  
  p
}

geom_basic_prc <- function(p, object, show_legend = TRUE, add_np_nn = TRUE,
                           xlim, ylim, ratio, ...) {
  
  pn_info <- get_pn_info(object)
  
  if (add_np_nn && pn_info$is_consistant) {
    main <- make_rocprc_title(object, "Precision-Recall")
  } else {
    main <- "Precision-Recall"
  }
  
  p <- p + ggplot2::geom_hline(yintercept = pn_info$prc_base, colour = "red",
                               linetype = 'dashed')
  p <- set_coords(p, xlim, ylim, ratio)
  p <- geom_basic(p, main, "Recall", "Precision", show_legend)
  
  p
}

set_coords <- function(p, xlim, ylim, ratio) {
  
  if (is.null(ratio))  {
    p <- p + ggplot2::coord_cartesian(xlim = xlim, ylim = ylim)
  } else {
    p <- p + ggplot2::coord_fixed(ratio = ratio, xlim = xlim, ylim = ylim)
  }
  
  p
}

# First plot ROC and then PRCs:
ctype='ROC'
colors = brewer.pal(length(n_pos),'Spectral') # modify this to your favorite color palette, number of datasets is the first argument
combined_curves_df = ggplot2::fortify(combined_curves)
combined_curves_df = subset(combined_curves_df, curvetype == ctype)
combined_aucs = auc(combined_curves)$aucs
p <- ggplot(combined_curves_df, aes_string(x = 'x', y = 'y',color = 'modname'))
p <- p + ggplot2::geom_line(size=1,na.rm = TRUE) + scale_color_manual(values=colors)
xlim <- attr(combined_curves[["rocs"]], "xlim")
ylim <- attr(combined_curves[["rocs"]], "ylim")
roc=geom_basic_roc(p, combined_curves, show_legend = T, add_np_nn = F,
                   curve_df = combined_curves_df, xlim = xlim, ylim = ylim, ratio = 1)+ #add labels of auROC:
  annotate('text',x=0.12, y=0.54,color=colors[1], label=round(combined_aucs[1],2), fontface=2)+ 
  annotate('text',x=0.2, y=0.68,color=colors[2], label=round(combined_aucs[3],2), fontface=2)+
  annotate('text',x=0.3, y=0.8,color=colors[3], label=round(combined_aucs[5],2), fontface=2)+
  annotate('text',x=0.45, y=0.88,color=colors[4], label=round(combined_aucs[7],2), fontface=2)
combined_curves_df = ggplot2::fortify(combined_curves)
ctype <- 'PRC'
combined_curves_df <- subset(combined_curves_df, curvetype == ctype)
p <- ggplot(combined_curves_df, aes_string(x = 'x', y = 'y',color = 'modname'))
p <- p + ggplot2::geom_line(size=1,na.rm = TRUE)+ scale_color_manual(values=colors)
xlim <- attr(combined_curves[["prcs"]], "xlim")
ylim <- attr(combined_curves[["prcs"]], "ylim")
prc =geom_basic_prc(p, combined_curves, show_legend =T, add_np_nn = F,
                    curve_df = combined_curves_df, xlim = xlim, ylim = ylim, ratio = 1)+#add labels of auPRC:
  annotate('text',x=0.1, y=0.15,color=colors[1], label=round(combined_aucs[2],2), fontface=2)+
  annotate('text',x=0.29, y=0.5,color=colors[2], label=round(combined_aucs[4],2), fontface=2)+
  annotate('text',x=0.61, y=0.62,color=colors[3], label=round(combined_aucs[6],2), fontface=2)+
  annotate('text',x=0.8, y=0.83,color=colors[4], label=round(combined_aucs[8],2), fontface=2)
g = (roc|prc)+plot_layout(ncol=2,guides = 'collect') & labs(colour='% positive class')&theme(legend.position = 'bottom')
g

