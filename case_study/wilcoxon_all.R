df <- read.csv("log_transform_update.csv")
library(ggplot2)
library(ggpubr)
library(gridExtra)

a <- ggboxplot(df, x = "object", y = "rough",
               xlab = FALSE,
               ylab = "Structural roughness",
               palette = "npg", group = "object",
               color = "object", shape = "object",
               legend = "none", add = "jitter",
               add.params = list(size = 3)
) + labs(title = "a") + 
  font("ylab", size = 20) + 
  font("title", size = 23, face = "bold") + 
  theme(axis.text = element_text(size = 21)) + 
  stat_compare_means(
    method = "wilcox.test",
    paired = FALSE,
    ref.group = NULL,
    label.x = 1.5,
    size = 6
)  

b <- ggboxplot(df, x = "object", y = "log_curv",
               xlab = FALSE,
               ylab = "log(Gaussian curvature)",
               palette = "npg", group = "object",
               color = "object", shape = "object",
               legend = "none", add = "jitter",
               add.params = list(size = 3)
) + labs(title = "b") + 
  font("ylab", size = 20) + 
  font("title", size = 23, face = "bold") + 
  theme(axis.text = element_text(size = 21)) + 
  stat_compare_means(
    method = "wilcox.test",
    paired = FALSE,
    ref.group = NULL,
    label.x = 1.5,
    size = 6
)

c <- ggboxplot(df, x = "object", y = "log_convex",
               xlab = FALSE,
               ylab = "log(Convex hull)",
               palette = "npg", group = "object",
               color = "object", shape = "object",
               legend = "none", add = "jitter",
               add.params = list(size = 3)
) + labs(title = "c") + 
  font("ylab", size = 20) + 
  font("title", size = 23, face = "bold") + 
  theme(axis.text = element_text(size = 21)) + 
  stat_compare_means(
    method = "wilcox.test",
    paired = FALSE,
    ref.group = NULL,
    label.x = 1.5,
    size = 6
)

d <- ggboxplot(df, x = "object", y = "log_alpha",
               xlab = FALSE,
               ylab = "log(Aplha shape)",
               palette = "npg", group = "object",
               color = "object", shape = "object",
               legend = "none", add = "jitter",
               add.params = list(size = 3)
) + labs(title = "d") + 
  font("ylab", size = 20) + 
  font("title", size = 23, face = "bold") + 
  theme(axis.text = element_text(size = 21)) + 
  stat_compare_means(
    method = "wilcox.test",
    paired = FALSE,
    ref.group = NULL,
    label.x = 1.5,
    size = 6
)

e <- ggboxplot(df, x = "object", y = "log_entropy",
               xlab = FALSE,
               ylab = "log(Shannon entropy)",
               palette = "npg", group = "object",
               color = "object", shape = "object",
               legend = "none", add = "jitter",
               add.params = list(size = 3)
) + labs(title = "e") + 
  font("ylab", size = 20) + 
  font("title", size = 23, face = "bold") + 
  theme(axis.text = element_text(size = 21)) + 
  stat_compare_means(
    method = "wilcox.test",
    paired = FALSE,
    ref.group = NULL,
    label.x = 1.5,
    size = 6
)

f <- ggboxplot(df, x = "object", y = "log_gmm",
               xlab = FALSE,
               ylab = "log(GMM)",
               palette = "npg", group = "object",
               color = "object", shape = "object",
               legend = "none", add = "jitter",
               add.params = list(size = 3)    
) + labs(title = "f") + 
  font("ylab", size = 20) + 
  font("title", size = 23, face = "bold") + 
  theme(axis.text = element_text(size = 21)) + 
  stat_compare_means(
    method = "wilcox.test",
    paired = FALSE,
    ref.group = NULL,
    label.x = 1.5,
    size = 6
)

grid.arrange(a, b, c, d, e, f, nrow = 2, ncol = 3)

