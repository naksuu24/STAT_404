library(MASS)

### DATA ###
dat <- read.csv("results.csv")
dat$R <- factor(dat$rep)                                   
dat$G <- factor(dat$gen_method)                            
dat$S <- factor(dat$size, 
                levels = sort(unique(dat$size))) 
dat$D <- factor(dat$loop_density,
                levels = sort(unique(dat$loop_density)))  
dat$A <- factor(dat$algorithm,
                levels = c("bfs", "dijkstra", "astar", "greedy", "random_walk"))
dat$maze <- factor(dat$maze_id)                              
dat$y_raw <- dat$time_s                               

### MODEL CHECKING ###
# RAW
fit_sub <- lm(y_raw ~ maze + A + A:S + A:D + A:G, data = dat)
mz.dat <- aggregate(y_raw ~ maze + R + G + S + D, data = dat, FUN = mean)
fit_whole <- lm(y_raw ~ R + G + S + D + S:D, data = mz.dat)

resid <- resid(fit_sub)
fitted <- fitted(fit_sub)
resid.whole <- resid(fit_whole)
fitted.whole <- fitted(fit_whole)

plot(fitted, resid, col = as.integer(dat$A), pch = 16, cex = 0.7,
     xlab = "Fitted", ylab = "Residual",
     main = paste("Subplot residuals vs fitted (raw)"))
abline(h = 0, lty = 2)
legend("topleft", legend = levels(dat$A), col = seq_along(levels(dat$A)),
       pch = 16, cex = 0.7, bty = "n")
dev.copy(png, "rf_raw.png", width = 800, height = 800, res = 120); dev.off()

qqnorm(resid, main = "Normal Q-Q: subplot residuals", pch = 16, cex = 0.6)
qqline(resid)
dev.copy(png, "qq_raw.png", width = 800, height = 800, res = 120); dev.off()

plot(fitted.whole, resid.whole, pch = 16, xlab = "Fitted (maze mean)", ylab = "Residual",
     main = "Maze-level residuals vs fitted")
abline(h = 0, lty = 2)
dev.copy(png, "rf_whole_raw.png", width = 800, height = 800, res = 120); dev.off()

qqnorm(resid.whole, main = "Normal Q-Q: whole-plot residuals", pch = 16, cex = 0.6)
qqline(resid.whole)
dev.copy(png, "qq_whole_raw.png", width = 800, height = 800, res = 120); dev.off()

# BOX-COX
fit_bc <- lm(y_raw ~ maze + A + A:S + A:D + A:G, data = dat)
bc <- boxcox(fit_bc, lambda = seq(-2, 2, 0.05))
dev.copy(png, "boxcox.png", width = 800, height = 800, res = 120); dev.off()

# TRANSFORMED
dat$y_t <- log(dat$y_raw)

fit_sub.t <- lm(y_t ~ maze + A + A:S + A:D + A:G, data = dat)
mz.dat.t <- aggregate(y_t ~ maze + R + G + S + D, data = dat, FUN = mean)
fit_whole.t <- lm(y_t ~ R + G + S + D + S:D, data = mz.dat.t)

resid.t <- resid(fit_sub.t)
fitted.t <- fitted(fit_sub.t)
resid.whole.t <- resid(fit_whole.t)
fitted.whole.t <- fitted(fit_whole.t)

plot(fitted.t, resid.t, col = as.integer(dat$A), pch = 16, cex = 0.7,
     xlab = "Fitted", ylab = "Residual",
     main = paste("Subplot residuals vs fitted (transformed)"))
abline(h = 0, lty = 2)
legend("topleft", legend = levels(dat$A), col = seq_along(levels(dat$A)),
       pch = 16, cex = 0.7, bty = "n")
dev.copy(png, "rf_t.png", width = 800, height = 800, res = 120); dev.off()

qqnorm(resid.t, main = "Normal Q-Q: subplot residuals", pch = 16, cex = 0.6)
qqline(resid.t)
dev.copy(png, "qq_t.png", width = 800, height = 800, res = 120); dev.off()

plot(fitted.whole.t, resid.whole.t, pch = 16, xlab = "Fitted (maze mean)", ylab = "Residual",
     main = "Maze-level residuals vs fitted")
abline(h = 0, lty = 2)
dev.copy(png, "rf_whole_t.png", width = 800, height = 800, res = 120); dev.off()

qqnorm(resid.whole.t, main = "Normal Q-Q: whole-plot residuals", pch = 16, cex = 0.6)
qqline(resid.whole.t)
dev.copy(png, "qq_whole_t.png", width = 800, height = 800, res = 120); dev.off()

### ANOVA ###
fit <- aov(y_t ~ R + G + S + D + S:D + A + S:A + D:A + G:A + Error(maze), data = dat)
cat("\n===== Split-plot ANOVA =====\n")
print(summary(fit))

interaction.plot(
  x.factor = dat$gen_method,       
  trace.factor = dat$algorithm, 
  response = dat$y_t,
  fun = mean,
  type = "b",
  pch = c(15, 17, 1, 19, 20),
  xlab = "Maze Generation Algorithm",
  ylab = "Time (s)",
  trace.label = "Algorithm"
)
dev.copy(png, "AG_intplot.png", width = 800, height = 800, res = 120); dev.off()

### COMPARISON ###
fit_A <- aov(y_t ~ maze + A, data = dat)
tk <- TukeyHSD(fit_A, "A")
tk$A

# RANK BY FACTOR
rank_by <- function(f) apply(tapply(dat$y_t, list(dat$A, f), mean), 2, rank)
cat("Algorithm rank by size:\n"); print(rank_by(dat$S))
cat("Algorithm rank by loop density:\n"); print(rank_by(dat$D))
cat("Algorithm rank by generation method:\n"); print(rank_by(dat$G))

# PATH QUALITY AND SUCCESS RATE
dat$path_ratio <- ifelse(dat$success == 1, dat$path_length / dat$opt_path_length, NA)
cat("Mean path length / optimal (1 = shortest path):\n")
print(round(tapply(dat$path_ratio, dat$A, mean, na.rm = TRUE), 3))
cat("Success rate:\n"); print(round(tapply(dat$success, dat$A, mean), 3))