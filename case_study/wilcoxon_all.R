df <- read.csv("log_transform_update.csv")
library(ggplot2)
library(ggpubr)
library(gridExtra)

a <- ggboxplot(df,
  x = "object", y = "rough",
  xlab = FALSE,
  ylab = "Structural roughness",
  palette = "npg", group = "object",
  color = "object", shape = "object",
  legend = "none", add = "jitter",
  add.params = list(size = 3)
) + labs(title = "a. Structural roughness") +
  font("ylab", size = 20) +
  font("title", size = 20, face = "bold") +
  theme(axis.text = element_text(size = 20)) +
  stat_compare_means(
    method = "wilcox.test",
    paired = FALSE,
    ref.group = NULL,
    label.x = 1.5,
    label.y.npc = "top",
    size = 6,
    label = "p.signif"
  )

b <- ggboxplot(df,
  x = "object", y = "log_curv",
  xlab = FALSE,
  ylab = "log(Gaussian curvature)",
  palette = "npg", group = "object",
  color = "object", shape = "object",
  legend = "none", add = "jitter",
  add.params = list(size = 3)
) + labs(title = "b. Gaussian curvature") +
  font("ylab", size = 20) +
  font("title", size = 20, face = "bold") +
  theme(axis.text = element_text(size = 20)) +
  stat_compare_means(
    method = "wilcox.test",
    paired = FALSE,
    ref.group = NULL,
    label.x = 1.5,
    label.y.npc = "top",
    size = 6,
    label = "p.signif"
  )

c <- ggboxplot(df,
  x = "object", y = "log_convex",
  xlab = FALSE,
  ylab = "log(Convex hull)",
  palette = "npg", group = "object",
  color = "object", shape = "object",
  legend = "none", add = "jitter",
  add.params = list(size = 3)
) + labs(title = "c. Convex hull") +
  font("ylab", size = 20) +
  font("title", size = 20, face = "bold") +
  theme(axis.text = element_text(size = 20)) +
  stat_compare_means(
    method = "wilcox.test",
    paired = FALSE,
    ref.group = NULL,
    label.x = 1.5,
    label.y.npc = "top",
    size = 6,
    label = "p.signif"
  )

d <- ggboxplot(df,
  x = "object", y = "log_alpha",
  xlab = FALSE,
  ylab = "log(Alpha shape)",
  palette = "npg", group = "object",
  color = "object", shape = "object",
  legend = "none", add = "jitter",
  add.params = list(size = 3)
) + labs(title = "d. Alpha shape") +
  font("ylab", size = 20) +
  font("title", size = 20, face = "bold") +
  theme(axis.text = element_text(size = 20)) +
  stat_compare_means(
    method = "wilcox.test",
    paired = FALSE,
    ref.group = NULL,
    label.x = 1.5,
    label.y.npc = "top",
    size = 6,
    label = "p.signif"
  )

e <- ggboxplot(df,
  x = "object", y = "log_entropy",
  xlab = FALSE,
  ylab = "log(Shannon entropy)",
  palette = "npg", group = "object",
  color = "object", shape = "object",
  legend = "none", add = "jitter",
  add.params = list(size = 3)
) + labs(title = "e. Shannon entropy") +
  font("ylab", size = 20) +
  font("title", size = 20, face = "bold") +
  theme(axis.text = element_text(size = 20)) +
  stat_compare_means(
    method = "wilcox.test",
    paired = FALSE,
    ref.group = NULL,
    label.x = 1.5,
    label.y.npc = "top",
    size = 6,
    label = "p.signif"
  )

f <- ggboxplot(df,
  x = "object", y = "log_gmm",
  xlab = FALSE,
  ylab = "log(GMM)",
  palette = "npg", group = "object",
  color = "object", shape = "object",
  legend = "none", add = "jitter",
  add.params = list(size = 3)
) + labs(title = "f. GMM") +
  font("ylab", size = 20) +
  font("title", size = 20, face = "bold") +
  theme(axis.text = element_text(size = 20)) +
  stat_compare_means(
    method = "wilcox.test",
    paired = FALSE,
    ref.group = NULL,
    label.x = 1.5,
    label.y.npc = "top",
    size = 6,
    label = "p.signif"
  )

grid.arrange(a, b, c, d, e, f, nrow = 2, ncol = 3)
